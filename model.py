import random
import math
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np
class CountryAgent(Agent):
    def __init__(self, model, culture_vector, difference_threshhold):
        super().__init__(model)
        self.culture = culture_vector  # List of 6 Hofstede-style values ∈ [0,1]
        self.difference_threshhold  = difference_threshhold

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        max_distance = self.model.max_grid_diagonal

        for neighbor in neighbors:
            # Compute normalized Euclidean distance
            dx = self.pos[0] - neighbor.pos[0]
            dy = self.pos[1] - neighbor.pos[1]
            euclidean_distance = math.sqrt(dx**2 + dy**2)
            distance_score_raw = 1 - (euclidean_distance / max_distance)
            distance_score = max(distance_score_raw, self.model.min_connectivity) 

            # Compute similarity (Axelrod-style): number of similar dimensions 
            similarities = [
                abs(self.culture[i] - neighbor.culture[i]) < self.difference_threshhold  
                for i in range(len(self.culture))
            ]
            similarity_score = sum(similarities) / len(self.culture)
            similarity_distance_score = (similarity_score + distance_score) / 2
            # Interaction chance proportional to similarity
            if random.random() < similarity_distance_score:
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
    def __init__(self, width=10, height=10, min_connectivity=0.2, use_distance=True, difference_threshhold=0.1, seed=None):
        super().__init__(seed=seed)
        
        self.culture_dimension_stats = [
                    {'name': 'PDI',  'mean': 59, 'std': 21.34, 'min': 11,  'max': 104},
                    {'name': 'IND',  'mean': 46, 'std': 23.39, 'min': 12,  'max': 91},
                    {'name': 'MAS',  'mean': 49, 'std': 19.96, 'min': 5,   'max': 110},
                    {'name': 'UAI',  'mean': 67, 'std': 22.92, 'min': 8,   'max': 112},
                    {'name': 'LTO',  'mean': 49, 'std': 22.57, 'min': 9,   'max': 100}
                    ]
        self.num_culture_dimensions = 5

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid = SingleGrid(width, height, torus=False)
        self.min_connectivity = min_connectivity
        self.use_distance = use_distance
            
        if self.grid.width > 1 or self.grid.height > 1:
            self.max_grid_diagonal = math.sqrt((self.grid.width - 1)**2 + (self.grid.height - 1)**2)
        else:
            self.max_grid_diagonal = 0.0 
            
        for x in range(width):
            for y in range(height):
                # Remove any agent already at this position to avoid warnings (Mesa 3.0 safe)
                for existing_agent in self.grid.get_cell_list_contents((x, y)):
                    existing_agent.remove()

                culture_vector = []
                for i in range(self.num_culture_dimensions):
                    stats = self.culture_dimension_stats[i]
                    raw_value = np.random.normal(loc=stats['mean'], scale=stats['std'])
                    denominator = stats['max'] - stats['min']
                    normalized_value = (raw_value - stats['min']) / denominator
                    clipped_value = np.clip(normalized_value, 0, 1)
                    culture_vector.append(clipped_value)
                
                agent = CountryAgent(self, culture_vector=culture_vector, difference_threshhold=difference_threshhold)
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
        total_similarity = 0
        comparisons = 0

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a = agents[i].culture
                b = agents[j].culture
                num_dimensions = len(a)
                sim = sum(1 - abs(a[k] - b[k]) for k in range(num_dimensions)) / num_dimensions
                total_similarity += sim
                comparisons += 1

        return total_similarity / comparisons if comparisons else 1.0

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step") 