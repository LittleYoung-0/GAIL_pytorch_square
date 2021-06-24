# LittleYoung0326在原版基础上更改了参数名称、主函数的结构
# LittleYoung0517更改环境、扩大地图、增加障碍物、保存GAIL训练参数
# LittleYoung0622更改文件，更改智能体初始位置

from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from Env_square import Envsquare
import os
from GAIL_algo import GAIL, Generator, Discriminator
from RL_algo import REINFORCE

def RL_train(RL_file_name, MaxEpisode, MaxStep):
    # train by RL
    print('——————————reinforcement learning & expert policy——————————')
    agent_RL = REINFORCE(ActionNum=5)
    if os.path.exists(RL_file_name):
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
        agent_RL.save_model(RL_file_name)

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
        print('step', step, 'agent_RL at', expert_state_list[step], 'agent_RL action', expert_action_list[step],
              'reward', reward)
        if done:
            break

    return expert_state_list, expert_action_list

def GAIL_train(GAIL_file_name, MaxEpisode, MaxStep, expert_state_list, expert_action_list):
    # train by GAIL
    print('——————————GAIL  & learnt policy——————————')
    agent_GAIL = GAIL(StateDim=2, ActionNum=5)
    if os.path.exists(GAIL_file_name):
        print('已有保存模型，下载相关参数')
        agent_GAIL.load_model(GAIL_file_name)
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
            agent_GAIL.train_G(state_list, action_list, action_log_prob_list, reward_list, expert_state_list,
                               expert_action_list)
        agent_GAIL.save_model(GAIL_file_name)

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
        print('step', step, 'agent_GAIL at', gail_state_list[step], 'agent_GAIL action', gail_action_list[step],
              'reward', reward)
        if done:
            break

    return gail_state_list, gail_action_list


if __name__ == '__main__':
    torch.set_num_threads(1)
    GlobalReward = 5  # the reward of environment
    env = Envsquare(16, GlobalReward)
    MaxEpisode = 10000    # max episode number
    MaxStep = 100   # max step number of every episode

    RL_file_name = './RL_square.pkl'
    GAIL_file_name = './GAIL_square.pkl'

    expert_state_list, expert_action_list = RL_train(RL_file_name, MaxEpisode, MaxStep)

    gail_state_list, gail_action_list = GAIL_train(GAIL_file_name, MaxEpisode, MaxStep, expert_state_list, expert_action_list)

    print('——————————expert policy VS learnt policy——————————')
    print('expert policy')
    for i in range(len(expert_state_list)):
        print('step', i, 'expert agent state', expert_state_list[i], 'expert agent action', expert_action_list[i])
    print('learnt policy')
    for i in range(len(gail_state_list)):
        print('step', i, 'gail agent state', gail_state_list[i], 'gail agent action', gail_action_list[i])
































