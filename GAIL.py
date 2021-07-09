# LittleYoung0326在原版基础上更改了参数名称、主函数的结构
# LittleYoung0517更改环境、扩大地图、增加障碍物、保存GAIL训练参数
# LittleYoung0622更改文件，更改智能体初始位置

import torch
from Env_square import Envsquare
import os
from GAIL_algo import GAIL, Generator, Discriminator
import numpy as np

def GAIL_train(GAIL_file_name, MaxEpisode, MaxStep, expert_state_list_up, expert_action_list_up, expert_state_list_down, expert_action_list_down):
    # train by GAIL
    print('——————————GAIL  & learnt policy——————————')
    agent_GAIL = GAIL(StateDim=2, ActionNum=5)
    if os.path.exists(GAIL_file_name):
        print('已有保存模型，下载相关参数')
        agent_GAIL.load_model(GAIL_file_name)
    else:
        print('无保存模型，从头开始训练！')
        for episode in range(MaxEpisode):
            # 由随机起始点位置，确定专家数据是哪条
            if_up = env.random_reset()
            if if_up==True:
                expert_state_list = expert_state_list_up
                expert_action_list = expert_action_list_up
            else:
                expert_state_list = expert_state_list_down
                expert_action_list = expert_action_list_down
            # 开始训练
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
            print('Imitate by GAIL, Episode', episode, 'average reward', episode_reward)
            agent_GAIL.train_D(state_list, action_list, expert_state_list, expert_action_list)
            agent_GAIL.train_G(state_list, action_list, action_log_prob_list, reward_list, expert_state_list,
                               expert_action_list)

        agent_GAIL.save_model(GAIL_file_name)

    # record learnt policy
    gail_state_list = []
    gail_action_list = []
    env.random_reset()
    for step in range(MaxStep):
        env.render()
        state = env.get_state()
        action, action_prob, action_log_prob = agent_GAIL.get_action(state)
        gail_state_list.append(state)
        gail_action_list.append(action)
        reward, done = env.step([action, 0])
        # print('step', step, 'agent_GAIL at', gail_state_list[step], 'agent_GAIL action', gail_action_list[step],
        #       'reward', reward)
        if done:
            break

    return gail_state_list, gail_action_list

if __name__ == '__main__':
    torch.set_num_threads(1)
    GlobalReward = 5  # the reward of environment
    env = Envsquare(16, GlobalReward)
    MaxEpisode = 14000    # max episode number
    MaxStep = 100   # max step number of every episode

    GAIL_file_name = './GAIL_square0708.pkl'

    # 专家数据
    expert_state_list_up = np.array([[7/16, 1/16], [0.5, 0.0625], [0.5, 0.125], [0.5625, 0.125], [0.625, 0.125], [0.6875, 0.125], [0.6875, 0.1875], [0.6875, 0.25], [0.6875, 0.3125], [0.6875, 0.375], [0.6875, 0.375], [0.6875, 0.375], [0.75, 0.375], [0.75, 0.4375], [0.75, 0.5], [0.75, 0.5625], [0.75, 0.625], [0.6875, 0.625], [0.6875, 0.6875], [0.625, 0.6875], [0.625, 0.75], [0.625, 0.8125], [0.5625, 0.8125], [0.5, 0.8125], [0.5, 0.875], [0.5, 0.875]])
    expert_action_list_up = np.array([1, 3, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 0, 3, 0, 3, 3, 0, 0, 3, 3, 0])
    expert_state_list_down = ([[3/16, 1/16],[3/16,2/16], [3/16, 3/16],[3/16, 4/16],[0.25, 0.25], [0.3125, 0.25], [0.375, 0.25], [0.375, 0.3125], [0.375, 0.375], [0.4375, 0.375], [0.4375, 0.4375], [0.4375, 0.5], [0.4375, 0.5625], [0.5, 0.5625], [0.5, 0.625], [0.5, 0.6875], [0.5, 0.75], [0.5, 0.8125], [0.5, 0.875]])
    expert_action_list_down = np.array([ 3,3,3, 1     ,1, 1, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 0])

    # 训练GAIL
    gail_state_list, gail_action_list = GAIL_train(GAIL_file_name, MaxEpisode, MaxStep, expert_state_list_up, expert_action_list_up, expert_state_list_down, expert_action_list_down)

































