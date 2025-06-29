import random
import math
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np

# Agent representing a country with a Hofstede cultural profile
class CountryAgent(Agent):
    def __init__(self,  model, culture_vector, difference_threshhold):
        super().__init__(model) #IF IT DOESTWORK GO BACK 2 MORE STEPS
        self._pos = None
        self.culture = culture_vector  # Vector of normalized Hofstede cultural dimensions
        self.difference_threshhold  = difference_threshhold # Max allowed difference to consider a dimension "similar"
        
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        
            
# The model simulating cultural diffusion in a spatial grid
class CulturalModel(Model):
    def __init__(self, width=10, height=10, min_connectivity=0.2, difference_threshhold=0.1, seed=None):
        super().__init__(seed=seed)

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
            
        # Create and place agents with randomly sampled cultural traits
        for x in range(width):
            for y in range(height):
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
                agent = CountryAgent(self, culture_vector=culture_vector, difference_threshhold=difference_threshhold)
                self.grid.place_agent(agent, (x, y))
                
        # For data that we use later we used Data Collection
        self.datacollector = DataCollector(
            model_reporters={"AverageCultureSimilarity": self.compute_average_similarity,
                            "UniqueProfiles": self.count_unique_profiles})
        # self.datacollector.collect(self)
    # function for two countries to interact with each other
    def interact(self, agent_a, agent_b):
        if agent_a is agent_b:
            return
        # calculate the distances between 2 countries
        dx = agent_a.pos[0] - agent_b.pos[0]
        dy = agent_a.pos[1] - agent_b.pos[1]
        euclidean_distance = math.sqrt(dx**2 + dy**2)
        normalized_distance = 1 - (euclidean_distance / self.max_grid_diagonal)
        # use the min_connectivity
        effective_distance_score = max(normalized_distance, self.min_connectivity)
        # calculate the similarities
        # Cultural similarity score (proportion of similar dimensions)
        differences = np.abs(np.array(agent_a.culture) - np.array(agent_b.culture))
        similarities = differences < agent_a.difference_threshhold
        similarity_score = np.mean(similarities)
        similarity_distance_score = (similarity_score + effective_distance_score) / 2

        if self.random.random() < similarity_distance_score:
            differing_indices = [i for i in range(len(agent_a.culture)) if not similarities[i]]
            if differing_indices:
                feature_to_copy = self.random.choice(differing_indices)
                  # use a learning rate of 1 (the difference between culteres needs to be inverted so we can add it)
                    # The reason we choose 1 (total copy of neighbors feature) is that otherwise the computing time is pretty slow
                    # For actual real world simulation one would reduce this, as the assumption that one culture adapts a cultural trait by 100%
                    # copying it seems unreasonable
                alpha = 1.0  
                delta = agent_b.culture[feature_to_copy] - agent_a.culture[feature_to_copy]
                agent_a.culture[feature_to_copy] += alpha * delta


                
    # Count the number of unique cultural profiles (rounded to 1 decimal)
    def count_unique_profiles(self):
        profiles = set()
        for agent in self.agents:
            rounded_profile = tuple(round(v, 1) for v in agent.culture)
            profiles.add(rounded_profile)
        return len(profiles)
    
    # Compute average pairwise similarity among all agents
    def compute_average_similarity(self):
        # Get all culture vectors into a single NumPy array
        culture_matrix = np.array([agent.culture for agent in self.agents])
        num_agents, num_dims = culture_matrix.shape

        if num_agents < 2:
            return 1.0

        # Use NumPy broadcasting to compute all pairwise differences at once.
        diffs = np.abs(culture_matrix[:, np.newaxis, :] - culture_matrix[np.newaxis, :, :])
        # Calculate similarities (1 - diff) and average over dimensions
        pair_sims = np.mean(1 - diffs, axis=2)
        # We only want the upper triangle of the matrix (to avoid double-counting9
        i, j = np.triu_indices(num_agents, k=1)
        # Calculate the mean of just those upper-triangle values.
        return np.mean(pair_sims[i, j])
    
    # Advance model by one time step
    def step(self):
        agent_list = list(self.agents)
        self.random.shuffle(agent_list)

        for agent in agent_list:
            # Choose a random partner from the same list
            other_agent = self.random.choice(agent_list)
            self.interact(agent, other_agent)