import numpy as np
import matplotlib.pyplot as plt
import heapq
import random


class BBox :
    def __init__(self, start_pos, end_pos ):
        self.listof_coverage_area = []
        self.start_pos = start_pos
        self.end_pos = end_pos
    def calculate_overlap_area(self, rect1, rect2):
            x1, y1, x2, y2 = rect1
            x3, y3, x4, y4 = rect2
            x_overlap = max(0, min(x2, x4) - max(x1, x3))
            y_overlap = max(0, min(y2, y4) - max(y1, y3))
            return x_overlap * y_overlap

    def swap_positions(self, list, pos1, pos2):
            first_ele = list.pop(pos1)
            second_ele = list.pop(pos2)
            list.insert(pos1, second_ele)
            list.insert(pos2, first_ele)
            return list



    def Run(self):
        for j in range(len(self.start_pos)):
            target_index = j
            target_start = self.start_pos[target_index]
            target_end = self.end_pos[target_index]
            target_bbox = [min(target_start[0], target_end[0]), min(target_start[1], target_end[1]),
                           max(target_start[0], target_end[0]), max(target_start[1], target_end[1])]

            total_coverage_area = 0

            for i in range(len(self.start_pos)):
                if i == target_index:
                    continue
                bbox_i = [min(self.start_pos[i][0], self.end_pos[i][0]), min(self.start_pos[i][1], self.end_pos[i][1]),
                          max(self.start_pos[i][0], self.end_pos[i][0]), max(self.start_pos[i][1], self.end_pos[i][1])]
                overlap_area = self.calculate_overlap_area(target_bbox, bbox_i)
                total_coverage_area += overlap_area

            self.listof_coverage_area.append(total_coverage_area)


        for i in range(len(self.start_pos)):
            for j in range(len(self.start_pos)-1):
                if self.listof_coverage_area[i] > self.listof_coverage_area[j]:
                    self.swap_positions(self.start_pos, i, j)
                    self.swap_positions(self.end_pos, i, j)
                    self.swap_positions(self.listof_coverage_area, i, j)

        #print(self.start_pos, "\n", self.end_pos, "\n", self.listof_coverage_area)
        return self.start_pos, self.end_pos

def goal_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class Astar:
    def __init__(self, map_size, start, goal):
        self.size = map_size
        self.path = None
        self.start = start
        self.current = self.start
        self.goal = goal
        # self.dqn = DQNAgent()

    def calculate_cost(self, g_score, obstacles, is_reroute):
        g_cost = g_score.get(self.current, float('inf'))
        g_cost = g_cost + 1
        h_cost = goal_distance(self.current, self.goal)
        f_cost = g_cost + h_cost
        if is_reroute and self.current in obstacles:
            g_cost += self.size ** 2
            f_cost += self.size ** 2
        return g_cost, f_cost

    def get_neighbors(self, point, obstacles, is_reroute):
        x, y = point
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if is_reroute:
            return [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.size
                    and 0 <= ny < self.size]
        else:
            return [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.size
                    and 0 <= ny < self.size and (nx, ny) not in obstacles]

    def reconstruct_path(self, came_from, cur):
        ac_path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            ac_path.append(cur)
        ac_path.reverse()
        self.path = ac_path
        return ac_path

    def astar(self, obstacles, is_reroute):
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {self.start: 0}

        g_score[self.start], f_score = self.calculate_cost(g_score, obstacles, is_reroute)
        heapq.heappush(open_set, (f_score, self.start))

        while open_set:
            current_f, self.current = heapq.heappop(open_set)

            if self.current == self.goal:
                return self.reconstruct_path(came_from, self.current)

            closed_set.add(self.current)

            for neighbor in self.get_neighbors(self.current, obstacles, is_reroute):
                if neighbor in closed_set:
                    continue

                tentative_g, f_score = self.calculate_cost(g_score, obstacles, is_reroute)

                if neighbor not in [item[1] for item in open_set] or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = self.current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (f_score, neighbor))

        self.current = self.start
        return None

    def delete_path(self):
        self.current = self.start
        self.path = None


class Expert:
    def __init__(self, num=5, map_size = 20):
        self.num = num
        self.policy = []
        self.map_size = map_size

    def instruct(self, starts, goals):
        mdict = {(-1, 0):0, (0, 1):1, (1, 0):2, (0, -1):3}
        self.policy = []
        obstacles = set()
        obstacles.update(starts + goals)
        start_pos, end_pos = BBox(starts,goals).Run()
        Agents = [Astar(self.map_size, start_pos[i], end_pos[i]) for i in range(self.num)]
        anum = 0
        shorts = {}

        while anum < self.num:
            if Agents[anum].astar(obstacles, False):
                #  成功繞到終點
                obstacles.update(Agents[anum].path)
                anum += 1
            else:
                #  無法繞到終點，需re_route
                #  增加牆壁的cost，並允許發生shorts
                Agents[anum].astar(obstacles, True)
                obstacles.update(Agents[anum].path)
                #  尋找shorts發生的位置
                for a in range(anum):
                    for p in Agents[anum].path:
                        if p in Agents[a].path:
                            shorts[p] = a
                #print('Shorts:', shorts)
                anum += 1
                #  將原路線刪除
                # Agents[anum].delete_path()

        path = [r.path for r in Agents]
        starts_p = [r.start for r in Agents]
        goal_p = [r.goal for r in Agents]
        movements = [[] for i in range(self.num)]
        for i, p in enumerate(path):
            for j in range(1, len(p)):
                movedir = (p[j][0]-p[j-1][0], p[j][1]-p[j-1][1])
                movements[i].append(mdict[movedir])

        l = len(max(path, key=len)) - 1
        fixed_path = [arr + [-1] * (l - len(arr)) for arr in movements]
        steps = list(zip(*fixed_path[::1]))
        for s in steps:
            self.policy += [x for x in s if x != -1]

        return starts_p, goal_p, shorts




def generate_coordinates(size=20, n=5):
    if n > size * size:  # 最多只能有400個不重複的座標點
        raise ValueError("Cannot generate more than 400 unique coordinates in a 20x20 grid.")

    all_coordinates = [(x, y) for x in range(20) for y in range(20)]
    sample_data = random.sample(all_coordinates, n * 2)
    return sample_data[:n], sample_data[n:]



def RoutePainter(grid_size, final_paths):
    gridsize = grid_size
    maze = np.zeros((gridsize, gridsize))
    plt.figure(figsize=(gridsize, gridsize))
    plt.imshow(maze, cmap='gray', interpolation='nearest')
    plt.xticks(range(gridsize))
    plt.yticks(range(gridsize))
    plt.grid(color='black', linewidth=1)

    # 繪製代理者 1 的最短路徑
    for i in range(len(final_paths)):
        print("PATH" + str(i) + ":", final_paths[i])
        if not final_paths[i]:
            continue
        path_x = [pos[0] for pos in final_paths[i]]
        path_y = [pos[1] for pos in final_paths[i]]
        plt.plot(path_x, path_y, linewidth=2, label='agent' + str(i))

    plt.legend()
    plt.show()

def switcher(ag_id, num_agents, cur_state):

    ag_id += 1
    if ag_id > num_agents:
        ag_id = 1

    while cur_state.getPos(ag_id) == cur_state.getGoal(ag_id):
        ag_id += 1
        if ag_id > num_agents:
            ag_id = 1

    return ag_id
