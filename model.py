import random
import math
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np

# Agent representing a country with a Hofstede cultural profile
class CountryAgent(Agent):
    def __init__(self, unique_id,  model, culture_vector, difference_threshhold):
        self.unique_id = unique_id
        self.model = model
        self._pos = None
        self.culture = culture_vector  # Vector of normalized Hofstede cultural dimensions
        self.difference_threshhold  = difference_threshhold # Max allowed difference to consider a dimension "similar"
        self.base_global_distance_score_raw = 0.0  # global proximity scores initialization
        
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        
    # One step in the simulation: try to culturally adapt from a neighboring agent
    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        effective_global_distance_score = max(self.base_global_distance_score_raw, self.model.min_connectivity)
        for neighbor in neighbors:
            # Compare each cultural dimension to the neighbor
            similarities = [
                abs(self.culture[i] - neighbor.culture[i]) < self.difference_threshhold  
                for i in range(len(self.culture))
            ]
            # Cultural similarity score (proportion of similar dimensions)
            similarity_score = sum(similarities) / len(self.culture)
            # Combine with geographic proximity into interaction likelihood
            similarity_distance_score = (similarity_score + effective_global_distance_score) / 2
            if self.model.random.random() < similarity_distance_score:
                # Get the list of differing cultural indicies that could be copied
                differing_indices = [
                    i for i in range(len(self.culture))
                    if not similarities[i]
                ]
                if differing_indices:
                    # Randomly choose a cultural feature to copy
                    feature_to_copy = self.model.random.choice(differing_indices) 
                    # use a learning rate of -1 (the difference between culteres needs to be inverted so we can add it)
                    # The reason we choose -1 (total copy of neighbors feature) is that otherwise the computing time is pretty slow
                    # For actual real world simulation one would reduce this, as the assumption that one culture adapts a cultural trait by 100%
                    # copying it seems unreasonable
                    alpha = -1
                    # difference between cultures time the learning rate
                    cultural_delta = (self.culture[feature_to_copy] - neighbor.culture[feature_to_copy]) 
                    # add the change to the own culture
                    self.culture[feature_to_copy] += cultural_delta * alpha

# The model simulating cultural diffusion in a spatial grid
class CulturalModel(Model):
    def __init__(self, width=10, height=10, min_connectivity=0.2, difference_threshhold=0.1, seed=None):
        super().__init__()
        
        if seed is not None:
            self.reset_randomizer(seed) 
            random.seed(seed)
            np.random.seed(seed)
        else:
            self.reset_randomizer(random.randrange(10**9))
            
        # Hofstede cultural dimension statistics taken from Malinoski, 2012
        self.culture_dimension_stats = [
                    {"name": "PDI",  "mean": 59, "std": 21.34, "min": 11,  "max": 104},
                    {"name": "IND",  "mean": 46, "std": 23.39, "min": 12,  "max": 91},
                    {"name": "MAS",  "mean": 49, "std": 19.96, "min": 5,   "max": 110},
                    {"name": "UAI",  "mean": 67, "std": 22.92, "min": 8,   "max": 112},
                    {"name": "LTO",  "mean": 49, "std": 22.57, "min": 9,   "max": 100}
                    ]
        
        self.num_culture_dimensions = len(self.culture_dimension_stats)
        # Initialize grid and connectivity settings
        self.grid = SingleGrid(width, height, torus=False)
        self.min_connectivity = min_connectivity
        
        # Precompute max possible distance on the grid for normalization
        if self.grid.width > 1 or self.grid.height > 1:
            self.max_grid_diagonal = math.sqrt((self.grid.width - 1)**2 + (self.grid.height - 1)**2)
        else:
            self.max_grid_diagonal = 0.0 
            
        agent_id_counter = 0
        # Create and place agents with randomly sampled cultural traits
        for x in range(width):
            for y in range(height):
                 # Clean up pre-existing agents in this cell (if any)
                current_agents_in_cell = list(self.grid.get_cell_list_contents((x,y))) 
                for existing_agent in current_agents_in_cell:
                    self.grid.remove_agent(existing_agent)
                # Create normalized cultural vector based on Hofstede stats
                culture_vector = []
                for i in range(self.num_culture_dimensions):
                    stats = self.culture_dimension_stats[i]
                    raw_value = self.random.normalvariate(mu=stats["mean"], sigma=stats["std"])
                    denominator = stats["max"] - stats["min"]
                    if denominator > 0:
                        normalized_value = (raw_value - stats["min"]) / denominator
                    else:
                        normalized_value = 0.5 
                    clipped_value = np.clip(normalized_value, 0, 1)
                    culture_vector.append(clipped_value)
                # Create agent and place it in the grid
                agent = CountryAgent(agent_id_counter, self, culture_vector=culture_vector, difference_threshhold=difference_threshhold)
                self.grid.place_agent(agent, (x, y))
                self.agents.add(agent)

                agent_id_counter += 1
        # Compute global proximity scores for all agents
        self._precompute_all_agents_global_distance_scores()
        # For data that we use later we used Data Collection
        self.datacollector = DataCollector(
            model_reporters={"AverageCultureSimilarity": self.compute_average_similarity,
                            "UniqueProfiles": self.count_unique_profiles})
        self.datacollector.collect(self)
        
    # Precompute geographic "connectedness" score based on the proximity for each agent
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
                
    # Count the number of unique cultural profiles (rounded to 1 decimal)
    def count_unique_profiles(self):
        profiles = set()
        for agent in self.agents:
            rounded_profile = tuple(round(v, 1) for v in agent.culture)
            profiles.add(rounded_profile)
        return len(profiles)
    # Compute average pairwise similarity among all agents
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
    # Advance model by one time step
    def step(self):
        self.agents.shuffle_do("step") 
        self.datacollector.collect(self)
