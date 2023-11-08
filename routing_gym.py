import gym
from gym import spaces
import numpy as np
from copy import deepcopy




class State(object):
    def __init__(self, starts, goals, wsize=20):
        self.agents = deepcopy(starts)
        self.goals = deepcopy(goals)
        self.paths = [[pt] for pt in starts]
        self.num_agents = len(goals)
        self.wsize = wsize
        self.move_list = ((-1, 0), (0, 1), (1, 0), (0, -1))


    def getObservation(self, ag_id):
        obs = np.stack((self.get_obstacle(ag_id), self.get_agents_pos(ag_id), self.get_neighbors_goal(ag_id),
                        self.get_agent_goal(ag_id)))
        return obs

    def getPos(self, ag_id):
        return self.agents[ag_id - 1]


    def getPath(self, ag_id):
        return self.paths[ag_id - 1]

    def getGoal(self, ag_id):
        return self.goals[ag_id - 1]

    def getDistance(self, ag_id):
        px, py = self.agents[ag_id - 1]
        gx, gy = self.goals[ag_id - 1]
        distance = (gx - px, gy - py)
        return distance

    def getObset(self):
        obs_set = set()
        for path in self.paths:
            obs_set.update(path)

        return obs_set

    def act(self, ag_id, action):

        """
        execute the action and return the status of the agent

        """
        index = ag_id - 1
        dxo, dyo = self.getDistance(ag_id)
        ag_move = self.move_list[action]
        ax = self.agents[index][0] + ag_move[0]
        ay = self.agents[index][1] + ag_move[1]
        next_pos = (ax, ay)
        reward = 0.0
        if next_pos == self.getGoal(ag_id):  # reach the goal
            self.agents[index] = next_pos
            reward = 20
        elif not (0 <= ax < 20 and 0 <= ay < 20):  # out of bound
            reward = -2
        elif next_pos in self.getObset():  # self collision
            reward = -2
        elif next_pos in self.goals:  # reach others' goals
            reward = -2
        else:
            self.agents[index] = next_pos
            dxn, dyn = self.getDistance(ag_id)
            if abs(dxn) + abs(dyn) - abs(dxo) - abs(dyo) == -1:
                reward = 0
            else:
                reward = -0.5

        self.paths[index].append(self.agents[index])

        return reward


    def done(self):
        return self.agents == self.goals

    def get_obstacle(self, id):
        gmap = np.ones((self.wsize, self.wsize))  # 初始化全為1的map

        for ag_id in range(1, self.num_agents + 1):
            for position in self.paths[ag_id - 1]:
                x, y = position
                if 0 <= x < self.wsize and 0 <= y < self.wsize:
                    gmap[x, y] = 0  # 已被走過的路設為0 (障礙物)

        # 获取代理为中心的9x9地图ㄖ
        agent_pos = self.getPos(id)
        central_map_size = 9
        central_map = np.zeros((central_map_size,central_map_size))
        vx = central_map_size //2
        start_x = agent_pos[0] - vx
        end_x = agent_pos[0]+vx+1
        start_y = agent_pos[1] -vx
        end_y = agent_pos[1] +vx+1
        for i in range(start_x,end_x) :
            for j in range(start_y, end_y) :
                # print(j,agent_pos[1])
                relative_x = i - agent_pos[0] + central_map_size // 2
                relative_y = j - agent_pos[1] + central_map_size // 2
                if 0 <= i < 20 and 0 <= j < 20 and gmap[i,j] == 1:
                    central_map[relative_x, relative_y] = 1
        # print("current_pos", agent_pos)
        # print(gmap)
        # print(central_map)
        return central_map.astype(np.float32)

    def get_agents_pos(self, id):
        central_map_size = 9
        # 以agent為中心的9X9觀察空間
        agent_pos = self.getPos(id)
        central_map = np.zeros((central_map_size, central_map_size))

        # 其他agent的位置標註為其id
        for agent_id in range(1, self.num_agents + 1):
            other_pos = self.getPos(agent_id)
            relative_x = other_pos[0] - agent_pos[0] + central_map_size // 2
            relative_y = other_pos[1] - agent_pos[1] + central_map_size // 2
            if 0 <= relative_x < central_map_size and 0 <= relative_y < central_map_size:
                central_map[relative_x, relative_y] = agent_id

        return central_map.astype(np.float32)

    def get_neighbors_goal(self, id):
        central_map_size = 9
        # 以agent為中心的9X9觀察空間
        agent_pos = self.getPos(id)
        central_map = np.zeros((central_map_size, central_map_size))

        # 其他agent的終點標註為其對應id
        for other_id in range(1, self.num_agents + 1):
            if other_id != id:
                other_goal = self.getGoal(other_id)
                relative_x = other_goal[0] - agent_pos[0] + central_map_size // 2
                relative_y = other_goal[1] - agent_pos[1] + central_map_size // 2
                if 0 <= relative_x < central_map_size and 0 <= relative_y < central_map_size:
                    central_map[relative_x, relative_y] = other_id

        return central_map.astype(np.float32)

    def get_agent_goal(self, id):
        central_map_size = 9
        # 以agent為中心的9X9觀察空間
        agent_pos = self.getPos(id)
        central_map = np.zeros((central_map_size, central_map_size))

        # 自己的終點標註為自己的id
        goal = self.getGoal(id)
        relative_x = goal[0] - agent_pos[0] + central_map_size // 2
        relative_y = goal[1] - agent_pos[1] + central_map_size // 2
        if 0 <= relative_x < central_map_size and 0 <= relative_y < central_map_size:
            central_map[relative_x, relative_y] = id

        return central_map.astype(np.float32)


class RoutingEnv(gym.Env):
    def __init__(self, starts, goals, wsize=20):
        self.wsize = 20
        self.state = State(starts, goals, wsize)
        self.num_agents = self.state.num_agents
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])

    def step(self, ag_id, action):
        rewards = self.state.act(ag_id, action)
        return self.get_observation(), rewards, self.state.done(), {}

    def get_observation(self):
        return self.state

    def reset(self, starts, goals):
        self.state = State(starts, goals)
        return self.state