import random
import math
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np
class CountryAgent(Agent):
    def __init__(self, model, pos, culture_vector):
        super().__init__(model)
        self.pos = pos
        self.culture = culture_vector  # List of 6 Hofstede-style values ∈ [0,1]

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        max_distance = math.sqrt((self.model.grid.width - 1) ** 2 + (self.model.grid.height - 1) ** 2)

        for neighbor in neighbors:
            # Compute normalized Euclidean distance
            dx = self.pos[0] - neighbor.pos[0]
            dy = self.pos[1] - neighbor.pos[1]
            euclidean_distance = math.sqrt(dx**2 + dy**2)
            distance_score = 1 - (euclidean_distance / max_distance)

            # Skip if distance filtering is active and this neighbor is too far
            if self.model.use_distance and distance_score < self.model.min_distance_threshold:
                continue

            # Compute similarity (Axelrod-style): number of similar dimensions
            similarities = [
                abs(self.culture[i] - neighbor.culture[i]) < 0.1  # consider similar if difference < 0.1
                for i in range(len(self.culture))
            ]
            similarity_score = sum(similarities) / len(self.culture)

            if similarity_score == 0:
                continue  # No chance to interact

            # Interaction chance proportional to similarity
            if random.random() < similarity_score:
                # Find differing features
                differing_indices = [
                    i for i in range(len(self.culture))
                    if not similarities[i]
                ]
                if differing_indices:
                    # Randomly pick one differing dimension to adopt from neighbor
                    feature_to_copy = random.choice(differing_indices)
                    self.culture[feature_to_copy] = neighbor.culture[feature_to_copy]


class CulturalModel(Model):
    def __init__(self, width=10, height=10, min_distance_threshold=0.2, use_distance=True, seed=None):
        super().__init__(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid = SingleGrid(width, height, torus=False)
        self.min_distance_threshold = min_distance_threshold
        self.use_distance = use_distance

        for x in range(width):
            for y in range(height):
                # Remove any agent already at this position to avoid warnings (Mesa 3.0 safe)
                for existing_agent in self.grid.get_cell_list_contents((x, y)):
                    existing_agent.remove()

                # Create a new agent with a random 6-dim cultural vector
                culture_vector = [random.uniform(0, 1) for _ in range(6)]
                agent = CountryAgent(self, pos=(x, y), culture_vector=culture_vector)
                self.grid.place_agent(agent, (x, y))
                
        self.datacollector = DataCollector(
            model_reporters={"AverageCultureSimilarity": self.compute_average_similarity,
                            "UniqueProfiles": self.count_unique_profiles})

    def count_unique_profiles(self):
        profiles = set()
        for agent in self.agents:
            # Round each dimension to 1 decimal place to group similar cultures
            rounded_profile = tuple(round(v, 1) for v in agent.culture)
            profiles.add(rounded_profile)
        return len(profiles)

       
        

    def compute_average_similarity(self):
        agents = list(self.agents)
        if len(agents) < 2:
            return 1.0

        total_similarity = 0
        comparisons = 0

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a = agents[i].culture
                b = agents[j].culture
                sim = sum(1 - abs(a[k] - b[k]) for k in range(6)) / 6
                total_similarity += sim
                comparisons += 1

        return total_similarity / comparisons if comparisons else 1.0

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")  # Mesa 3.0: activates all agents randomly each step