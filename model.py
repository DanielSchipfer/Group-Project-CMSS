import math
import random
import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_space_component
from mesa.space import NetworkGrid
import networkx as nx


class CountryAgent(Agent):
    def __init__(self, model, culture_vector, position):
        super().__init__(model)
        self.culture_vector = culture_vector
        self.position = position  # (lat, lon)

    def step(self):
        neighbors = list(self.model.grid.get_neighbors(self.pos)) 
        if not neighbors:
            return

        interaction_partner = self.model.random.choice(neighbors)
        similarity = self.compute_similarity(interaction_partner)

        if self.model.random.random() < similarity:
            diff_indices = [i for i, (a, b) in enumerate(zip(self.culture_vector, interaction_partner.culture_vector)) if a != b]
            if diff_indices:
                chosen_index = self.model.random.choice(diff_indices)
                self.culture_vector[chosen_index] = interaction_partner.culture_vector[chosen_index]


    def compute_similarity(self, other_agent):
        vec_diff = np.linalg.norm(np.array(self.culture_vector) - np.array(other_agent.culture_vector))
        culture_similarity = 1 - (vec_diff / (len(self.culture_vector) * 100))  # scale to [0,1]
        geo_distance = self.geographic_distance(self.position, other_agent.position)
        geo_weight = math.exp(-geo_distance / self.model.distance_decay)
        return culture_similarity * geo_weight

    def geographic_distance(self, pos1, pos2):
        # Simple haversine formula
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        radius = 6371  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius * c


class CulturalDiffusionModel(Model):
    def __init__(self, n_agents=10, seed=None, distance_decay=1000):
        super().__init__(seed=seed)
        self.n_agents = n_agents
        self.distance_decay = distance_decay
        self.datacollector = DataCollector(model_reporters={"Steps": lambda m: m.steps})
        self.G = nx.Graph()
        self.grid = NetworkGrid(self.G)

        self.agent_list = []

        # Generate agents and nodes
        for i in range(n_agents):
            culture_vector = [self.random.randint(0, 100) for _ in range(6)]
            position = (self.random.uniform(-90, 90), self.random.uniform(-180, 180))
            agent = CountryAgent(self, culture_vector, position)
            self.G.add_node(i)  # use index as node ID
            self.G.nodes[i]["agent"] = []  # initialize the agent list for this node
            self.grid.place_agent(agent, i)  # place agent at node i
            self.agent_list.append(agent)


        # Connect all nodes to all others (fully connected)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                self.G.add_edge(i, j)


    def step(self):
        self.random.shuffle(self.agent_list)
        for agent in self.agent_list:
            agent.step()
        self.datacollector.collect(self)


