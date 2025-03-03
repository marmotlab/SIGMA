import numpy as np
import heapq
import random
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, n, num_agents):
        self.n = n
        self.maze = np.random.choice([0, 1], size=(n, n), p=[0.7, 0.3])
        self.num_agents = num_agents
        self.agents = []
        self.generate_agents()

    def generate_agents(self):
        for _ in range(self.num_agents):
            start, goal = self.random_position(), self.random_position()
            while self.maze[start] == 1 or self.maze[goal] == 1 or start == goal:
                start, goal = self.random_position(), self.random_position()
            self.agents.append((start, goal))

    def random_position(self):
        return random.randint(0, self.n - 1), random.randint(0, self.n - 1)

    def display_maze(self):
        plt.imshow(self.maze, cmap='gray')
        for agent in self.agents:
            start, goal = agent
            plt.scatter(start[1], start[0], c='red', marker='o')
            plt.scatter(goal[1], goal[0], c='blue', marker='x')
        plt.show()

class ODrMStar:
    def __init__(self, maze, agents):
        self.maze = maze
        self.agents = agents
        self.paths = {agent: [] for agent in agents}

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, pos):
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = []
        for dx, dy in neighbors:
            x2, y2 = pos[0] + dx, pos[1] + dy
            if 0 <= x2 < self.maze.shape[0] and 0 <= y2 < self.maze.shape[1] and self.maze[x2, y2] == 0:
                result.append((x2, y2))
        return result

    def astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.neighbors(current):
                tentative_g_score = current_g + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
        return []

    def plan_paths(self):
        for start, goal in self.agents:
            path = self.astar(start, goal)
            self.paths[(start, goal)] = path
        return self.paths

# Example usage
n = 10
num_agents = 3

# Generate maze and agents
maze = Maze(n, num_agents)
maze.display_maze()

# Plan paths using ODrMStar
odrm_star = ODrMStar(maze.maze, maze.agents)
paths = odrm_star.plan_paths()

# Display paths
print("Paths for agents:")
for agent, path in paths.items():
    print(f"Agent {agent}: Path {path}")
