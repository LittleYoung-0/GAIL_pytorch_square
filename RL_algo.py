import torch
from GAIL_algo import Generator
import os

class REINFORCE(object):
    def __init__(self, ActionNum):
        self.ActionNum = ActionNum
        self.generator = Generator(self.ActionNum)

    def get_action(self, observ):
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