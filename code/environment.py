# -*- coding: UTF-8 -*-
import numpy as np
import copy

class IoT:
    CLOCK = 0
    ENERGY = 80

    def __init__(self):
        """
        # 环境状态空间 -- 7
        1. 环境需要该节点完成其本职工作
        2. 环境需要该节点参与设备协作
        3. 环境需要其使用服务，完成某种操作
        4. 环境需要其对使用的服务等待反馈
        5. 环境中存在恶意节点  --设备发现恶意行为
        """

        self.observation_space = np.array([
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1],
        ])
        self.states_num = self.observation_space.shape[1]  # state 状态空间个数
        self.n_states = self.observation_space.shape[1]  # 一个 state 维度

        self.state = self.observation_space[np.random.randint(
            0, self.n_states+1)].copy()
        # 终止状态
        self.terminate_state = np.array([0, 0, 0, 0, 0])

        """action_space:
        6 个动作
        
        0. 全 0 什么也不做，代表补充能源
        1. 做自己的本职工作
        2. 与他人协作
        3. 请求服务
        4. 接受服务，并对服务提供反馈
        5. 检测并报告异常行为 
        """

        self.action_space = np.array([0, 1, 2, 3, 4, 5])
        self.n_actions = self.action_space.shape[0]

        self.job_delta = 15
        self.job_threshold = 5
        self.service_delta = 20
        self.service_threshold = 10

    def resetEnv(self):
        """
        重置环境的状态，返回观察。如果回合结束，就要调用此函数，重置环境信息
        """
        self.CLOCK = 0
        self.ENERGY = 50
        a = np.random.randint(0, self.states_num)
        self.state = self.observation_space[a].copy()
        return self.state.copy()

    def clock_energy(self, action):
        if action == 0:
            return 0, 0
        else:
            if action == 1:
                return 1, -1
            elif action == 2:
                return 1, -1
            elif action == 3:
                return 1, -3
            elif action == 4:
                return 1, -1
            elif action == 5:
                return 3, -3

    @staticmethod
    def malicious_confirm_reward() -> int:
        possibility = [0, 0, 0, 1, 1, 1, 2, 2, 3]
        return possibility[np.random.randint(0, len(possibility))]

    def next_state_reward(self, action):  # when the action is done in right situation/state
        if action == 1:   # 做本职工作
            delay = (self.CLOCK-1) % self.job_delta
            if delay <= self.job_threshold:
                reward = 1 - 0.2 * delay
            else:
                reward = self.job_threshold - delay

            self.state[0] = 0
            next_state = self.state.copy()
        elif action == 2:  # 与他人协作
            reward = 1
            self.state[1] = 0
            next_state = self.state.copy()
        elif action == 3:  # 请求服务
            reward = 1
            self.state[2], self.state[3] = 0, 1
            next_state = self.state.copy()
        elif action == 4:  # 接受服务，并对服务进行反馈
            delay = (self.CLOCK-1) % self.job_delta
            if delay <= self.service_threshold:
                reward = 1 - 0.2 * delay
            else:
                reward = self.service_threshold - delay
            self.state[3] = 0
            next_state = self.state.copy()
        elif action == 5:  # 检测报告异常行为
            reward = self.malicious_confirm_reward()
            self.state[4] = 0
            next_state = self.state.copy()

        return next_state, reward

    def step(self, action):

        # 判断系统当前状态是否为终止状态
        if (self.state == self.terminate_state).all():
            return self.state, 0, True

        clock, energy = self.clock_energy(action)

        if action == 0 or self.ENERGY < -energy: # action = [0,0,0,0]，补充能源

            self.CLOCK += 1
            self.ENERGY = 50
            return self.state.copy(), 0, False

        self.CLOCK += clock
        self.ENERGY += energy

        index = np.nonzero(self.state)[0]
        if action-1 in index:  # 存在动作对应恰当的状态
            next_state, reward = self.next_state_reward(action)
        else:
            next_state, reward = self.state.copy(), 0

        return next_state, reward, False
