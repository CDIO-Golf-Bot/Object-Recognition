import heapq

class PathFinder:
    def __init__(self, config, grid_manager):
        self.config = config
        self.grid = grid_manager

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neigh = (current[0] + dx, current[1] + dy)
                if (0 <= neigh[0] <= self.grid.real_w // self.grid.spacing and
                    0 <= neigh[1] <= self.grid.real_h // self.grid.spacing and
                    neigh not in self.grid.obstacles):
                    tentative = g_score[current] + 1
                    if tentative < g_score.get(neigh, float('inf')):
                        came_from[neigh] = current
                        g_score[neigh] = tentative
                        f = tentative + self.heuristic(neigh, goal)
                        heapq.heappush(open_set, (f, neigh))
        return []