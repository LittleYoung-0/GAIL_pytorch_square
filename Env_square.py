import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import cv2

class Envsquare(object):
    def __init__(self, size, GlobalReward):
        self.map_size = size
        self.GlobalReward = GlobalReward
        self.raw_occupancy = np.zeros((self.map_size, self.map_size))
        for i in range(self.map_size):
            # 边界障碍
            self.raw_occupancy[0][i] = 1
            self.raw_occupancy[self.map_size - 1][i] = 1
            self.raw_occupancy[i][0] = 1
            self.raw_occupancy[i][self.map_size - 1] = 1
        # 障碍是按照16 * 16的地图尺寸设计的
        # 中间的障碍
        self.raw_occupancy[0:4, int((self.map_size - 1) / 2)] = 1
        self.raw_occupancy[8:12, int((self.map_size - 1) / 2)] = 1
        # 左边的障碍
        self.raw_occupancy[4:8, int((self.map_size - 1) / 4)] = 1
        self.raw_occupancy[12:15, int((self.map_size - 1) / 4)] = 1
        # 右边的障碍
        self.raw_occupancy[4:8, int((self.map_size - 1) * 3 / 4)] = 1
        self.raw_occupancy[12:15, int((self.map_size - 1) * 3 / 4)] = 1

        # 中间障碍的上下两部分是空的
        # self.raw_occupancy[1][int((self.map_size - 1) / 2)] = 0
        # self.raw_occupancy[self.map_size - 2][int((self.map_size - 1) / 2)] = 0

        self.occupancy = self.raw_occupancy.copy()

        self.agt1_pos = [int((self.map_size - 1) / 2), 1]
        self.goal1_pos = [int((self.map_size - 1) / 2), self.map_size - 2]
        self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1

    def reset(self):
        self.occupancy = self.raw_occupancy.copy()

        self.agt1_pos = [int((self.map_size - 1) / 2 -3), 8]    # 智能体起始位置
        self.goal1_pos = [int((self.map_size - 1) / 2), self.map_size - 2]
        self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1

    def get_state(self):
        state = np.zeros((1, 2))
        state[0, 0] = self.agt1_pos[0] / self.map_size
        state[0, 1] = self.agt1_pos[1] / self.map_size
        return state

    def step(self, action_list):
        reward = 0
        # agent1 move
        if action_list[0] == 0:  # move up
            if self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] - 1
                self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
        elif action_list[0] == 1:  # move down
            if self.occupancy[self.agt1_pos[0] + 1][self.agt1_pos[1]] != 1:  # if can move
                self.agt1_pos[0] = self.agt1_pos[0] + 1
                self.occupancy[self.agt1_pos[0] - 1][self.agt1_pos[1]] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
        elif action_list[0] == 2:  # move left
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] != 1:  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] - 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1
        elif action_list[0] == 3:  # move right
            if self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] + 1] != 1:  # if can move
                self.agt1_pos[1] = self.agt1_pos[1] + 1
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1] - 1] = 0
                self.occupancy[self.agt1_pos[0]][self.agt1_pos[1]] = 1

        if self.agt1_pos == self.goal1_pos:
            reward = reward + self.GlobalReward

        done = False
        if reward == self.GlobalReward:
            done = True
        return reward, done

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 0] = 1.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 1] = 0.0
        obs[self.agt1_pos[0], self.agt1_pos[1], 2] = 0.0
        # 加了显示终点的代码LY
        obs[self.goal1_pos[0],self.goal1_pos[1], 0] = 0.0
        obs[self.goal1_pos[0], self.goal1_pos[1], 1] = 1.0
        obs[self.goal1_pos[0], self.goal1_pos[1], 2] = 0.0
        return obs

    def render(self):
        obs = self.get_global_obs()
        enlarge = 30
        new_obs = np.ones((self.map_size * enlarge, self.map_size * enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):   # 根据通道来上色

                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge),
                                  (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge),
                                  (0, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge),
                                  (0, 255, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge),
                                  (255, 0, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(100)

