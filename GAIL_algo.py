import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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

        expert_state = torch.Tensor(expert_state_list).float()
        expert_action = self.int_to_tensor(expert_action_list[0])
        for i in range(1, len(expert_state_list)):
            # temp = torch.Tensor(expert_state_list[i]).float()
            # expert_state = torch.cat([expert_state, temp], dim=0)
            temp = self.int_to_tensor(expert_action_list[i])
            expert_action = torch.cat([expert_action, temp], dim=0)


        agent_label = torch.zeros(len(state_list), 1)
        expert_label = torch.ones(len(expert_state_list), 1)

        expert_predict = self.discrimnator(expert_state, expert_action)
        loss = self.advantage_loss(expert_predict, expert_label)

        agent_predict = self.discrimnator(agent_state, agent_action)
        loss = loss + self.advantage_loss(agent_predict, agent_label)    # 这里的loss是生成器和判别器两个loss相加
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

        # expert_state = torch.from_numpy(expert_state_list[0]).float()
        # expert_action = self.int_to_tensor(expert_action_list[0])
        # for i in range(1, len(expert_state_list)):
        #     temp = torch.from_numpy(expert_state_list[i]).float()
        #     expert_state = torch.cat([expert_state, temp], dim=0)
        #     temp = self.int_to_tensor(expert_action_list[i])
        #     expert_action = torch.cat([expert_action, temp], dim=0)

        expert_state = torch.Tensor(expert_state_list).float()
        expert_action = self.int_to_tensor(expert_action_list[0])
        for i in range(1, len(expert_state_list)):
            # temp = torch.Tensor(expert_state_list[i]).float()
            # expert_state = torch.cat([expert_state, temp], dim=0)
            temp = self.int_to_tensor(expert_action_list[i])
            expert_action = torch.cat([expert_action, temp], dim=0)

        agent_predict = self.discrimnator(agent_state, agent_action)
        fake_reward = agent_predict.mean()

        agent_loss = torch.FloatTensor([0.0])
        for i in range(len(state_list)):
            agent_loss = agent_loss + fake_reward * action_log_prob_list[i]
        agent_loss = -agent_loss/len(state_list)


        self.g_optimizer.zero_grad()
        agent_loss.backward()
        self.g_optimizer.step()

    def save_model(self, file_name):
        torch.save(self.generator, file_name)

    def load_model(self, file_name):
        self.generator = torch.load(file_name)