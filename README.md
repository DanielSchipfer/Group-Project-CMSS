# Group-Project-CMSS

Model:
 One Agent is one country Behavior: 
• Agents interact with each other and adapt a cultural feature (Hofstede cultural dimension, one value that contains them all) based on the similarity and proximity to the other Agents Variables:
 • Hofstede cultural dimensions for one Agent gets drawn from a normal distribution
 • distance between Agents takes a grid like in Axelrod, where the distance between two Agents is calculated, value between 0-1 where 1 means they are completely adjacent to each other and 0 means the max distance. 
o When 2 Agents interact a similarity_score and distance_score gets calculated based on Hofstede cultural dimensions and the distance between 2 Agents

Research Question:
How does varying the baseline level of interconnectedness (represented by min_connectivity) influence the speed and extent of cultural convergence, where this baseline ensures a minimum chance of interaction even for geographically distant countries due to factors like ICT?
