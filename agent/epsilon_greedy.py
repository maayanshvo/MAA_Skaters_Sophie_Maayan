import random

def initialize_values(list_of_actions, init_mode = 0):
    return [1 for i in range(len(list_of_actions))]

class EpsilonGreedy():

	def __init__(self, epsilon, list_of_actions, init_value = 0):
		self.epsilon = epsilon
		self.list_of_actions = list_of_actions
		self.avg_values = [random.uniform(init_value, init_value + 0.5) for i in range(len(list_of_actions))]
		self.counts = [0 for i in range(len(list_of_actions))]
	

	def selectAction(self):
		prob = random.uniform(0, 1)
		if prob > self.epsilon and not (len(set(self.avg_values)) <= 1):
			return self.list_of_actions[self.avg_values.index(max(self.avg_values))]
		else:
			return random.choice(self.list_of_actions)

	def update(self, action_index, reward):
		self.counts[action_index] = self.counts[action_index] + 1
		old_val = self.avg_values[action_index]
		increment = (reward - old_val) / self.counts[action_index]
		new_val = old_val + increment
		self.avg_values[action_index] = new_val
	





