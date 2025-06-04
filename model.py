import mesa
import numpy as np
import random

def calculate_similarity(value1, value2):
    """Calculate similarity between two Hofstede cultural scores.
    Use inverse of absolute distance normalized to [0,1]."""
    max_distance = 10  # Assumed max difference scale of scores
    dist = abs(value1 - value2)
    similarity = max(0, 1 - dist / max_distance)
    return similarity

def normalized_grid_distance(pos1, pos2, width, height):
    """Calculate normalized distance between two grid positions in [0,1]. 
    1 means adjacent, 0 means max distance."""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    max_dist = np.sqrt(width ** 2 + height ** 2)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    norm_dist = 1 - (dist / max_dist)  # Invert: closer = higher value
    return norm_dist

class CountryAgent(mesa.Agent):
    def __init__(self, model, pos, culture_score=None, country_name=None):
        super().__init__(model)
        self.pos = pos
        # Initialize culture score randomly if not provided
        self.culture_score = culture_score if culture_score is not None else random.uniform(0, 10)
        self.country_name = country_name

    def step(self):
        neighbor_agents = self.model.grid.get_neighbors(self.pos, moore=False, include_center=False)
        if not neighbor_agents:
            return

        neighbor_agent = self.random.choice(neighbor_agents)

        sim = calculate_similarity(self.culture_score, neighbor_agent.culture_score)
        dist_score = normalized_grid_distance(self.pos, neighbor_agent.pos, self.model.width, self.model.height)

        if dist_score < self.model.min_distance_threshold:
            # If distance is below threshold, no interaction
            return

        if self.model.use_distance_in_similarity:
            interaction_prob = sim * dist_score
        else:
            interaction_prob = sim

        if self.random.random() < interaction_prob:
            diff = neighbor_agent.culture_score - self.culture_score
            step_size = 0.1
            self.culture_score += step_size * diff
            self.culture_score = np.clip(self.culture_score, 0, 10)

class CulturalDiffusionModel(mesa.Model):
    def __init__(self, width=10, height=10, seed=None, min_distance_threshold=0.0, use_distance_in_similarity=True):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.min_distance_threshold = min_distance_threshold
        self.use_distance_in_similarity = use_distance_in_similarity

        self.grid = mesa.space.SingleGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents and place on grid
        for x in range(width):
            for y in range(height):
                agent = CountryAgent(self, pos=(x, y))
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.running = True

    def step(self):
        self.schedule.step()

    def get_average_cultural_similarity(self):
        total_sim = 0
        count = 0
        for agent in self.schedule.agents:
            neighbor_agents = self.grid.get_neighbors(agent.pos, moore=False, include_center=False)
            for neighbor in neighbor_agents:
                sim = calculate_similarity(agent.culture_score, neighbor.culture_score)
                total_sim += sim
                count += 1
        return total_sim / count if count > 0 else 0

    def is_stable(self):
        for agent in self.schedule.agents:
            neighbor_agents = self.grid.get_neighbors(agent.pos, moore=False, include_center=False)
            for neighbor in neighbor_agents:
                sim = calculate_similarity(agent.culture_score, neighbor.culture_score)
                dist_score = normalized_grid_distance(agent.pos, neighbor.pos, self.width, self.height)
                if dist_score >= self.min_distance_threshold and sim < 1.0:
                    return False
        return True
