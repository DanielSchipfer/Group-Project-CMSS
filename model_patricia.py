import random
import numpy as np

class AxelrodModel:
    def __init__(self, size = 10, F = 5, q = 15, seed = None):
        self.size = size
        self.F = F
        self.q = q
        self.grid = np.zeros((size, size, F), dtype=int)

        if seed is not None:
            random.seed(seed % (2**32))
            np.random.seed(seed % (2**32))

        self.grid = np.random.randint(0, q, size = (size, size, F))

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = (x + dx) % self.size, (y + dy) % self.size
            neighbors.append((nx, ny))
        return neighbors

    def step(self):
        x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
        neighbors = self.get_neighbors(x, y)
        nx, ny = random.choice(neighbors)

        agent = self.grid[x, y]
        neighbor = self.grid[nx, ny]

        #similarity
        similarity = np.sum(agent == neighbor) / self.F

        if 0 < similarity < 1:
            if random.random() < similarity:
                diff_features = np.where(agent != neighbor)[0]
                if len(diff_features) > 0:
                    feature = random.choice(diff_features)
                    self.grid[x, y, feature] = neighbor[feature]

    def run(self, steps = 100000):
        for _ in range(steps):
            self.step()

    def count_regions(self):
        culture_ids = np.sum(self.grid * (self.q ** np.arange(self.F)), axis = 2)
        unique_ids = np.unique(culture_ids)
        region_map = np.zeros_like(culture_ids)
        count = 0
        for cid in unique_ids:
            mask = (culture_ids == cid)
            labeled, num = label(mask)
            count += num
        return count

    def plot_culture_map(self, ax = None, title = None):
        culture_ids = np.sum(self.grid * (self.q ** np.arange(self.F)), axis=2)
        if ax is None:
            fig, ax = plt.subplots(figsize = (6, 6))
            
        ax.imshow(culture_ids, cmap='tab20', interpolation='nearest')
        
        if title:
            ax.set_title(title)
        ax.axis('off')