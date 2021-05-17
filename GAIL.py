# LittleYoung0326在原版基础上更改了参数名称、主函数的结构
# LittleYoung0517更改环境、扩大地图、增加障碍物、保存GAIL训练参数
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from Env_square import Envsquare
import os

class Generator(nn.Module):
    def __init__(self, ActionNum):
        super(Generator, self).__init__()
        self.ActionNum = ActionNum
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, self.ActionNum)

    def get_action(self, input):
        # 使用functional里的relu，可独立使用
        # 区别nn中的relu，只能在nn的继承环境下使用：https://blog.csdn.net/landing_guy_/article/details/114498511
        temp = F.relu(self.layer1(input))
        temp = F.relu(self.layer2(temp))
        action_prob = F.softmax(self.layer3(temp), dim=1)    # 神经网络的输出值，对每一列进行softmax,每个动作的概率值
        # 根据概率值对其采样，squeeze移除一个维度，Categorical：https://blog.csdn.net/ProQianXiao/article/details/102893824
        m = Categorical(action_prob.squeeze(0))
        action = m.sample()
        # 计算样本的对数概率
        action_log_prob =m.log_prob(action)
        return action.item(), action_prob, action_log_prob

class Discriminator(nn.Module):
    def __init__(self, StateDim, ActionNum):      # StateDim:状态维度
        super(Discriminator, self).__init__()
        self.StateDim = StateDim
        self.ActionNum = ActionNum
        self.layer1 = nn.Linear(self.StateDim + self.ActionNum, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # 一个前向传播
        state_action = torch.cat([state, action], 1)
        output = torch.relu(self.layer1(state_action))
        output = torch.relu(self.layer2(output))
        output = torch.sigmoid(self.layer3(output))
        return output

class GAIL(object):
    def __init__(self, StateDim, ActionNum):
        self.StateDim = StateDim
        self.ActionNum = ActionNum
        self.generator = Generator(self.ActionNum)
        self.discrimnator = Discriminator(self.StateDim, self.ActionNum)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        self.d_optimizer = torch.optim.Adam(self.discrimnator.parameters(), lr=1e-3)
        # 下面两个loss函数有什么区别？有什么用？
        self.loss = torch.nn.MSELoss()
        self.advantage_loss = torch.nn.BCELoss()
        self.Gamma = 0.9

    def get_action(self, observ):
        action, action_prob, action_log_prob = self.generator.get_action(torch.from_numpy(observ).float())
        return action, action_prob, action_log_prob

    def int_to_tensor(self, action):
        temp = torch.zeros(1, self.ActionNum)
        temp[0, action] = 1
        return temp

    def train_D(self, state_list, action_list, expert_state_list, expert_action_list):
        # torch.from_numpy():https://blog.csdn.net/weixin_36670529/article/details/110293613
        agent_state = torch.from_numpy(state_list[0]).float()
        agent_action = self.int_to_tensor(action_list[0])   # 为什么这里要自定义函数转tensor？
        for i in range(1, len(state_list)):
            temp = torch.from_numpy(state_list[i]).float()
            agent_state = torch.cat([agent_state, temp], dim=0)   # 按行拼接
            temp = self.int_to_tensor(action_list[i])
            agent_action = torch.cat([agent_action, temp], dim=0)

        expert_state = torch.from_numpy(expert_state_list[0]).float()
        expert_action = self.int_to_tensor(expert_action_list[0])
        for i in range(1, len(expert_state_list)):
            temp = torch.from_numpy(expert_state_list[i]).float()
            expert_state = torch.cat([expert_state, temp], dim=0)
            temp = self.int_to_tensor(expert_action_list[i])
            expert_action = torch.cat([expert_action, temp], dim=0)

        agent_label = torch.zeros(len(state_list), 1)
        expert_label = torch.ones(len(expert_state_list), 1)

        expert_predict = self.discrimnator(expert_state, expert_action)
        loss = self.advantage_loss(expert_predict, expert_label)

        agent_predict = self.discrimnator(agent_state, agent_action)
        loss = loss + self.advantage_loss(agent_predict, agent_label)    # 这里的loss函数为什么这样计算？
        # print("Chen SHOW Discriminator LOSS: ", loss)
        # torch网络前向传播：https://blog.csdn.net/scut_salmon/article/details/82414730
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

    def train_G(self, state_list, action_list, action_log_prob_list, reward_list, expert_state_list, expert_action_list):
        agent_state = torch.from_numpy(state_list[0]).float()
        agent_action = self.int_to_tensor(action_list[0])
        for i in range(1, len(state_list)):
            temp = torch.from_numpy(state_list[i]).float()
            agent_state = torch.cat([agent_state, temp], dim=0)
            temp = self.int_to_tensor(action_list[i])
            agent_action = torch.cat([agent_action, temp], dim=0)

        expert_state = torch.from_numpy(expert_state_list[0]).float()
        expert_action = self.int_to_tensor(expert_action_list[0])
        for i in range(1, len(expert_state_list)):
            temp = torch.from_numpy(expert_state_list[i]).float()
            expert_state = torch.cat([expert_state, temp], dim=0)
            temp = self.int_to_tensor(expert_action_list[i])
            expert_action = torch.cat([expert_action, temp], dim=0)

        agent_predict = self.discrimnator(agent_state, agent_action)
        fake_reward = agent_predict.mean()

        agent_loss = torch.FloatTensor([0.0])
        for i in range(len(state_list)):
            agent_loss = agent_loss + fake_reward * action_log_prob_list[i]
        agent_loss = -agent_loss/len(state_list)
        # print("Chen SHOW Generator LOSS: ", agent_loss)

        self.g_optimizer.zero_grad()
        agent_loss.backward()
        self.g_optimizer.step()

    def save_model(self):
        torch.save(self.generator, 'GAIL_square.pkl')

    def load_model(self):
        self.generator = torch.load('GAIL_square.pkl')

class REINFORCE(object):
    def __init__(self, ActionNum):
        self.ActionNum = ActionNum
        self.generator = Generator(self.ActionNum)

    def get_action(self, observ):   # 这个observ是什么？
        action, action_prob, action_log_prob = self.generator.get_action(torch.from_numpy(observ).float())
        return action, action_prob, action_log_prob

    def train(self, action_list, action_prob_list, reward_list):
        agent_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        # 计算某轨迹下全部动作的折扣奖励
        T = len(reward_list)
        Gain_list = torch.zeros(1, T)
        Gain_list[0, T - 1] = torch.FloatTensor([reward_list[T - 1]])
        for i in range(T - 2, -1, -1):
            Gain_list[0, i] = reward_list[i] + 0.95 * Gain_list[0, i + 1]

        agent_loss = torch.FloatTensor([0.0])
        for t in range(T):
            agent_loss = agent_loss + Gain_list[0, t] * torch.log(action_prob_list[t][0, action_list[t]])  # action_prob_list是什么形状？
        agent_loss = -agent_loss / T
        agent_optimizer.zero_grad()
        agent_loss.backward()
        agent_optimizer.step()

    def save_model(self):
        torch.save(self.generator, 'RL_square.pkl')

    def load_model(self):
        self.generator = torch.load('RL_square.pkl')

if __name__ == '__main__':
    torch.set_num_threads(1)
    GlobalReward = 5  # the reward of environment
    env = Envsquare(16, GlobalReward)
    MaxEpisode = 40000    # max episode number
    MaxStep = 100   # max step number of every episode

    # train by RL
    print('——————————reinforcement learning & expert policy——————————')
    agent_RL = REINFORCE(ActionNum=5)
    if os.path.exists('./RL_square.pkl'):
        print('已有保存模型，下载相关参数')
        agent_RL.load_model()
    else:
        print('无保存模型，从头开始训练！')
        for episode in range(MaxEpisode):
            env.reset()
            action_list = []
            action_prob_list = []
            reward_list = []
            episode_reward = 0
            for step in range(MaxStep):
                # env.render()
                state = env.get_state()
                action, action_prob, action_log_prob = agent_RL.get_action(state)
                action_list.append(action)
                action_prob_list.append(action_prob)
                reward, done = env.step([action, 0])
                episode_reward = episode_reward + reward
                reward_list.append(reward)
                if done:
                    break
            print('Train expert, Episode:', episode, 'Step:', step, 'every step reward:', episode_reward / step)
            if done:
                agent_RL.train(action_list, action_prob_list, reward_list)
        agent_RL.save_model()

    # record expert policy
    expert_state_list = []
    expert_action_list = []
    env.reset()
    for step in range(MaxStep):
        env.render()
        state = env.get_state()
        action, action_prob, action_log_prob = agent_RL.get_action(state)
        expert_state_list.append(state)
        expert_action_list.append(action)
        reward, done = env.step([action, 0])
        print('step', step, 'agent_RL at', expert_state_list[step], 'agent_RL action', expert_action_list[step], 'reward', reward)
        if done:
            break

    # train by GAIL
    print('——————————GAIL  & learnt policy——————————')
    agent_GAIL = GAIL(StateDim=2, ActionNum=5)
    if os.path.exists('./GAIL_square.pkl'):
        print('已有保存模型，下载相关参数')
        agent_GAIL.load_model()
    else:
        print('无保存模型，从头开始训练！')
        for episode in range(MaxEpisode):
            env.reset()
            state_list = []
            action_list = []
            reward_list = []
            action_log_prob_list = []
            episode_reward = 0
            for step in range(MaxStep):
                # env.render()
                state = env.get_state()
                action, action_prob, action_log_prob = agent_GAIL.get_action(state)
                state_list.append(state)
                action_list.append(action)
                action_log_prob_list.append(action_log_prob)
                reward, done = env.step([action, 0])
                episode_reward = episode_reward + reward
                reward_list.append(reward)
                if done:
                    break
            print('Imitate by GAIL, Episode', episode, 'average reward', episode_reward / step)
            agent_GAIL.train_D(state_list, action_list, expert_state_list, expert_action_list)
            agent_GAIL.train_G(state_list, action_list, action_log_prob_list, reward_list, expert_state_list, expert_action_list)
        agent_GAIL.save_model()

    # record learnt policy
    gail_state_list = []
    gail_action_list = []
    env.reset()
    for step in range(MaxStep):
        env.render()
        state = env.get_state()
        action, action_prob, action_log_prob = agent_GAIL.get_action(state)
        gail_state_list.append(state)
        gail_action_list.append(action)
        reward, done = env.step([action, 0])
        print('step', step, 'agent_GAIL at', gail_state_list[step], 'agent_GAIL action', gail_action_list[step], 'reward', reward)
        if done:
            break

    print('——————————expert policy VS learnt policy——————————')
    print('expert policy')
    for i in range(len(expert_state_list)):
        print('step', i, 'expert agent state', expert_state_list[i], 'expert agent action', expert_action_list[i])
    print('learnt policy')
    for i in range(len(gail_state_list)):
        print('step', i, 'gail agent state', gail_state_list[i], 'gail agent action', gail_action_list[i])

































