import random
import math
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np
class CountryAgent(Agent):
    def __init__(self, unique_id,  model, culture_vector, difference_threshhold):
        self.unique_id = unique_id
        self.model = model
        self._pos = None
        self.culture = culture_vector  
        self.difference_threshhold  = difference_threshhold
        self.base_global_distance_score_raw = 0.0 
        
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        
    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        effective_global_distance_score = max(self.base_global_distance_score_raw, self.model.min_connectivity)
        for neighbor in neighbors:
            similarities = [
                abs(self.culture[i] - neighbor.culture[i]) < self.difference_threshhold  
                for i in range(len(self.culture))
            ]
            similarity_score = sum(similarities) / len(self.culture)
            similarity_distance_score = (similarity_score + effective_global_distance_score) / 2
            if self.model.random.random() < similarity_distance_score:
                differing_indices = [
                    i for i in range(len(self.culture))
                    if not similarities[i]
                ]
                if differing_indices:
                    feature_to_copy = self.model.random.choice(differing_indices) 
                    self.culture[feature_to_copy] = neighbor.culture[feature_to_copy]


class CulturalModel(Model):
    def __init__(self, width=10, height=10, min_connectivity=0.2, use_distance=True, difference_threshhold=0.1, seed=None):
        super().__init__()
        
        if seed is not None:
            self.reset_randomizer(seed) 
            random.seed(seed)
            np.random.seed(seed)
        else:
            self.reset_randomizer(random.randrange(10**9))
            
        self.culture_dimension_stats = [
                    {'name': 'PDI',  'mean': 59, 'std': 21.34, 'min': 11,  'max': 104},
                    {'name': 'IND',  'mean': 46, 'std': 23.39, 'min': 12,  'max': 91},
                    {'name': 'MAS',  'mean': 49, 'std': 19.96, 'min': 5,   'max': 110},
                    {'name': 'UAI',  'mean': 67, 'std': 22.92, 'min': 8,   'max': 112},
                    {'name': 'LTO',  'mean': 49, 'std': 22.57, 'min': 9,   'max': 100}
                    ]
        
        self.num_culture_dimensions = len(self.culture_dimension_stats)

        self.grid = SingleGrid(width, height, torus=False)
        self.min_connectivity = min_connectivity
        self.use_distance = use_distance
            
        if self.grid.width > 1 or self.grid.height > 1:
            self.max_grid_diagonal = math.sqrt((self.grid.width - 1)**2 + (self.grid.height - 1)**2)
        else:
            self.max_grid_diagonal = 0.0 
            
        agent_id_counter = 0
        for x in range(width):
            for y in range(height):
                current_agents_in_cell = list(self.grid.get_cell_list_contents((x,y))) 
                for existing_agent in current_agents_in_cell:
                    self.grid.remove_agent(existing_agent)

                culture_vector = []
                for i in range(self.num_culture_dimensions):
                    stats = self.culture_dimension_stats[i]
                    raw_value = self.random.normalvariate(mu=stats['mean'], sigma=stats['std'])
                    denominator = stats['max'] - stats['min']
                    if denominator > 0:
                        normalized_value = (raw_value - stats['min']) / denominator
                    else:
                        normalized_value = 0.5 
                    clipped_value = np.clip(normalized_value, 0, 1)
                    culture_vector.append(clipped_value)

                agent = CountryAgent(agent_id_counter, self, culture_vector=culture_vector, difference_threshhold=difference_threshhold)
                self.grid.place_agent(agent, (x, y))
                self.agents.add(agent)

                agent_id_counter += 1

        self._precompute_all_agents_global_distance_scores()
        self.datacollector = DataCollector(
            model_reporters={"AverageCultureSimilarity": self.compute_average_similarity,
                            "UniqueProfiles": self.count_unique_profiles})
        self.datacollector.collect(self)
        
    def _precompute_all_agents_global_distance_scores(self):
            if not self.agents or self.max_grid_diagonal == 0:
                for agent in self.agents:
                    agent.base_global_distance_score_raw = 1.0 
                return

            all_agents = list(self.agents) 
            num_total_agents = len(all_agents)

            if num_total_agents <= 1:
                if num_total_agents == 1:
                    all_agents[0].base_global_distance_score_raw = 1.0 
                return

            for agent in all_agents:
                total_distance_to_others = 0
                for other_agent in all_agents:
                    if agent == other_agent:
                        continue

                    dx = agent.pos[0] - other_agent.pos[0]
                    dy = agent.pos[1] - other_agent.pos[1]
                    euclidean_distance = math.sqrt(dx**2 + dy**2)
                    total_distance_to_others += euclidean_distance

                average_distance_to_others = total_distance_to_others / (num_total_agents - 1)

                agent.base_global_distance_score_raw = 1 - (average_distance_to_others / self.max_grid_diagonal)
                agent.base_global_distance_score_raw = max(0, min(1, agent.base_global_distance_score_raw))
                
    def count_unique_profiles(self):
        profiles = set()
        for agent in self.agents:
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
        self.agents.shuffle_do("step") 
        self.datacollector.collect(self)
