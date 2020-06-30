# https://github.com/ImparaAI/monte-carlo-tree-search

import random
from math import log, sqrt
import numpy as np
import math
import copy
import heapq
from utils import ucs, calculate_best_delivery_time, combine_probabilities


# class State:

# 	def __init__(self, robot_pose, time, tasks):
# 		self.robot_pose = robot_pose
# 		self.time = time
# 		self.tasks = tasks

class MCTS_Node:

	def __init__(self, id, pose_id, time, observations, requests_left_to_deliver, requests_delivered):
		self.id = id
		self.pose_id = pose_id
		self.time = time
		self.observations = observations
		self.requests_left_to_deliver = requests_left_to_deliver
		self.requests_delivered = requests_delivered
		self.expanded = False
		self.children = []
		self.unexplored_children_indices = []
		self.visits = 0
		self.num_expansions = 0
		self.cumulative_reward = 0.0


		# self.state = State(robot_pose, time, tasks)
		# self.action_taken = action_taken
		# self.node_reward = reward
		

		# self.win_value = 0
		# self.policy_value = None
		# self.parent = None
		# self.children = []
		# self.unexplored_indices = []
		
		# self.player_number = None
		# self.discovery_factor = 0.35
		# self.discovery_factor = 0.70710678118		# sqrt(2)/2
		## self.discovery_factor = 1.41421356237		# sqrt(2)
		# self.discovery_factor = 2.82842712475		# 2 sqrt(2)

	# def update_win_value(self, value):
	# 	self.win_value += value
	# 	self.visits += 1

	# 	if self.parent:
	# 		self.parent.update_win_value(value)

	# def update_cum_future_reward(self, value):
	# 	self.cum_future_reward += value
	# 	self.visits += 1
	# 	if self.parent:
	# 		self.parent.update_cum_future_reward(value)

	# def update_policy_value(self, value):
	# 	self.policy_value = value

	# def add_child(self, child):
	# 	self.children.append(child)
	# 	child.parent = self

	# def add_children(self, children):
	# 	for child in children:
	# 		self.add_child(child)

	# def get_best_child(self):
	# 	best_children = []
	# 	best_score = float('-inf')

	# 	for i in range(len(self.children)):
	# 		child = self.children[i]
	# 		score = child.get_score(self)

	# 		if score > best_score:
	# 			best_score = score
	# 			best_children = [i]
	# 		elif score == best_score:
	# 			best_children.append(i)

	# 	return random.choice(best_children)


	# def choose_child_for_exploration(self):

	# 	# if any children unexplored, choose an unexplored child at random
	# 	if len(self.unexplored_indices) != 0:
	# 		unexplored_index = random.choice(self.unexplored_indices)
	# 		self.unexplored_indices.remove(unexplored_index)

	# 		return unexplored_index

	# 	# otherwise choose child according to UCT
	# 	scores = []
	# 	for child in self.children:
	# 		scores.append(child.get_score(self))
	# 	scores = np.array(scores)
	# 	scores = scores/float(sum(scores))
	# 	rand_val  = random.random()

	# 	scores = list(scores)
	# 	cum_score = 0.0
	# 	for i in range(len(scores)):
	# 		cum_score += scores[i]
	# 		if cum_score > rand_val:
	# 			return i

	# def expected_reward(self):
	# 	if self.visits >= 1:
	# 		return self.cumulative_reward/self.visits
	# 	else:
	# 		return simulate()




	# def discount(reward):
	# 	if discount_strategy == 'multiplicative':
	# 		return discount_factor * reward

	# def exploration_score():
	# 	if exploration_strategy == 'ucb':
	# 		return ucb(expected_reward, num_parent_visits, num_visits)

	# def ucb(expected_reward, num_parent_visits, num_visits):
	# 	discovery_operand = discovery_factor * sqrt(log(num_parent_visits) / (num_visits or 1))
	# 	return expected_reward + discovery_operand


	# def get_score(self, parent_node):
	# 	# discovery_operand = self.discovery_factor * (self.policy_value or 1) * sqrt(log(parent_node.visits) / (self.visits or 1))
	# 	discovery_operand = self.discovery_factor * sqrt(log(parent_node.visits) / (self.visits or 1))

	# 	# win_multiplier = 1 if self.parent.player_number == root_node.player_number else -1
	# 	# win_operand = win_multiplier * self.win_value / (self.visits or 1)

	# 	ave_expected_reward = self.cum_future_reward/ (self.visits or 1)

	# 	# self.score = win_operand + discovery_operand
	# 	self.score = ave_expected_reward + discovery_operand

	# 	return self.score

	# def is_scorable(self):
	# 	return self.visits or self.policy_value != None








class MCTS:

	def __init__(self, spatial_graph, avails, observations, requests_left_to_deliver, start_pose, start_time, budget, max_iterations, planning_horizon, maintenance_reward, deliver_reward, mu, discovery_factor, distribution_node, maintenance_node):
		self.nodes = {}
		self.spatial_graph = spatial_graph
		self.avails = avails
		self.root_node_id = self.id_form(start_pose, start_time, observations, requests_left_to_deliver)
		self.start_time = start_time
		self.budget = budget
		self.max_iterations = max_iterations
		self.planning_horizon = planning_horizon
		self.maintenance_reward = maintenance_reward
		self.deliver_reward = deliver_reward
		self.mu = mu
		self.discovery_factor = discovery_factor
		self.distribution_node = distribution_node
		self.maintenance_node = maintenance_node
		self.nodes[self.root_node_id] = MCTS_Node(self.root_node_id, start_pose, start_time, observations, requests_left_to_deliver, [])


	def observations_standard_form(self, observations):
		pose_ids = list(self.spatial_graph)
		obs = ''
		for pose_id in pose_ids:
			if pose_id in observations:
				pose_obs_string = '_' + str(observations[pose_id][0]) + str(observations[pose_id][1])
				obs += pose_obs_string
			else:
				obs += '_3' 
		return obs

	def requests_left_to_deliver_standard_form(self, requests_left_to_deliver):
		pose_ids = list(self.spatial_graph)
		req = ''
		for pose_id in pose_ids:
			if pose_id in requests_left_to_deliver:
				req += '_1'
			else:
				req += '_0'
		return req

	def id_form(self, curr_node, curr_time, observations, requests_left_to_deliver):
		obs = self.observations_standard_form(observations)
		req = self.requests_left_to_deliver_standard_form(requests_left_to_deliver)
		return str(curr_node) + '_' + str(curr_time) + '_' +  obs + '_' +  req


	def exploration_score(self, expected_reward, num_parent_visits, num_visits):
		# ucb
		discovery_operand = self.discovery_factor * sqrt(log(num_parent_visits) / (num_visits or 1))
		return expected_reward + discovery_operand

	def create_policy(self):
		for iteration in range(self.max_iterations):
			reward = self.expand(self.root_node_id, self.planning_horizon-1, 0)
			# self.root_node = node

	def expand(self, node_id, planning_horizon, maintenance_reward_collected):
		node = self.nodes[node_id]
		if node.expanded == False:
			self.populate_children(node_id)
			node.expanded = True

		if (planning_horizon > 0) and (node.time < self.budget):
			child_id = self.choose_action_for_exploration(node_id, maintenance_reward_collected)
			if node.children[child_id][0] == 'maintenance':
				maintenance_reward_collected += self.maintenance_reward
			next_state = self.simulate_action(node_id, child_id)
			reward = expand(next_state, planning_horizon-1, maintenance_reward_collected)
			# cumulative_future_reward += future_reward
			node.visits += 1
			node.num_expansions += 1
			node.cumulative_reward += reward

		else:
			reward = self.rollout(node_id, maintenance_reward_collected)
			node.visits += 1
			node.cumulative_reward += reward

		self.nodes[node_id] = node
		return reward

	def calculate_expected_future_delivery_reward(self, node_id):
		node = self.nodes[node_id]
		expected_reward = 0.0
		for request in node.requests_left_to_deliver:
			last_observation = None
			if request in node.observations.keys():
				last_observation = node.observations[request]
			best_prob = calculate_best_delivery_time(self.avails[request], last_observation, node.time + ucs(self.spatial_graph, node.pose_id, request), self.start_time+self.budget, self.mu)
			expected_reward += best_prob*self.deliver_reward
		return expected_reward


	def rollout(self, node_id, maintenance_reward_collected):
		node = self.nodes[node_id]
		delivery_reward = len(node.requests_delivered)*self.deliver_reward
		expected_future_delivery_reward = self.calculate_expected_future_delivery_reward(node_id)
		return delivery_reward + expected_future_delivery_reward + maintenance_reward_collected

	def sample_occupancy(self, pose_id, deliver_time, observations):
		a_priori_prob = self.avails[pose_id].get_prediction(deliver_time)
		if pose_id in observations:
			last_observation = observations[pose_id][0]
			last_observation_time = observations[pose_id][1]
			prob = combine_probabilities(a_priori_prob, self.mu, deliver_time, last_observation, last_observation_time)
		else:
			prob = a_priori_prob

		if random.random() < prob:
			return 1
		else:
			return 0


	def simulate_action(self, node_id, child):
		node = self.nodes[node_id]
		child = node.children[child_id]

		if child_id in node.unexplored_children_indices:
			node.unexplored_children_indices.remove(child_index)
		node.children[child_id][1] += 1

		action = child[0]
		next_states = child[2]
		if action == 'move':
			next_state = next_states[0]

		if action == 'maintenance':
			next_state = next_states[0]

		if action == 'observe':
			available = self.sample_occupancy(node.pose_id, node.time, node.observations)
			if available:
				next_state = next_states[0]
			else:
				next_state = next_states[1]

		if action == 'deliver':
			deliver_time = node.time + 2*ucs(node.pose_id, self.distribution_node)
			available = self.sample_occupancy(node.pose_id, deliver_time, node.observations)
			if available:
				next_state = next_states[0]
			else:
				next_state = next_states[1]


		self.nodes[node_id] = node


	def expected_reward(self, node_id, maintenance_reward_collected):
		node = self.nodes[node_id]
		if node.visits >= 1:
			return node.cumulative_reward/node.visits
		else:
			return self.rollout(node_id, maintenance_reward_collected)


	def choose_best_action(self, node_id, min_expansions, maintenance_reward_collected):
		node = self.nodes[node_id]
		if (node.num_expansions < min_expansions) or not(len(node.unexplored_children_indices) == 0):
			return None 	# REPLAN
		else:
			best_score = -float("inf")
			best_child_index = None
			for i in range(len(children)):
				child = children[i]
				action = child[0]
				num_visits = child[1]
				next_states = child[2]

				if action == 'move':
					future_state = next_states[0]
					expected_reward = self.expected_reward(future_state, maintenance_reward_collected)
					score = expected_reward
					if score > best_score:
						best_score = score
						best_child_index = i

				if action == 'maintenance':
					future_state = next_states[0]
					expected_reward = self.expected_reward(future_state, maintenance_reward_collected)
					score = expected_reward
					if score > best_score:
						best_score = score
						best_child_index = i

				if action == 'observe':
					available = next_states[0]
					unavailable = next_states[1]

					if node.pose_id in node.observations:
						last_observation_value = last_observation[0]
						last_observation_time = last_observation[1]
						a_priori_prob = self.avails[node.pose_id].get_prediction(node.time)
						avail_prob = combine_probabilities(a_priori_prob, self.mu, node.time, last_observation_value, last_observation_time)
					else:
						avail_prob = self.avails[node.pose_id].get_prediction(node.time)

					expected_reward = avail_prob*self.expected_reward(available, maintenance_reward_collected) + (1.0 - avail_prob)*self.expected_reward(unavailable, maintenance_reward_collected)
					score = expected_reward
					if score > best_score:
						best_score = score
						best_child_index = i

				if action == 'deliver':
					available = next_states[0]
					unavailable = next_states[1]

					if node.pose_id in node.observations:
						last_observation_value = last_observation[0]
						last_observation_time = last_observation[1]
						a_priori_prob = self.avails[node.pose_id].get_prediction(self.states[available].time) 			# FIXME +1 minute to do delivery
						avail_prob = combine_probabilities(a_priori_prob, self.mu, self.states[available].time, last_observation_value, last_observation_time)
					else:
						avail_prob = self.avails[node.pose_id].get_prediction(self.states[available].time)

					expected_reward = avail_prob*self.expected_reward(available, maintenance_reward_collected) + (1.0 - avail_prob)*self.expected_reward(unavailable, maintenance_reward_collected)
					score = expected_reward
					if score > best_score:
						best_score = score
						best_child_index = i

			return best_child_index

	def choose_action_for_exploration(self, node_id, maintenance_reward_collected):
		node = self.nodes[node_id]

		# if any child unexplored
		if not(len(node.unexplored_children_indices) == 0):
			child_index = random.choice(node.unexplored_children_indices)
			best_child_index = child_index

		else:
			best_score = -float("inf")
			best_child_index = None
			for i in range(len(node.children)):
				action = node.children[i][0]
				num_visits = node.children[i][1]
				next_states = node.children[i][2]

				if action == 'move':
					future_state = next_states[0]
					expected_reward = self.expected_reward(future_state, maintenance_reward_collected)
					score = self.exploration_score(expected_reward, node.visits, num_visits)
					if score > best_score:
						best_score = score
						best_child_index = i

				if action == 'maintenance':
					future_state = next_states[0]
					expected_reward = self.expected_reward(future_state, maintenance_reward_collected)
					score = self.exploration_score(expected_reward, node.visits, num_visits)
					if score > best_score:
						best_score = score
						best_child_index = i

				if action == 'observe':
					available = next_states[0]
					unavailable = next_states[1]

					if node.pose_id in node.observations:
						last_observation_value = last_observation[0]
						last_observation_time = last_observation[1]
						a_priori_prob = self.avails[node.pose_id].get_prediction(node.time)
						avail_prob = combine_probabilities(a_priori_prob, self.mu, node.time, last_observation_value, last_observation_time)
					else:
						avail_prob = self.avails[node.pose_id].get_prediction(node.time)

					expected_reward = avail_prob*self.expected_reward(available, maintenance_reward_collected) + (1.0 - avail_prob)*self.expected_reward(unavailable, maintenance_reward_collected)
					score = self.exploration_score(expected_reward, node.visits, num_visits)
					if score > best_score:
						best_score = score
						best_child_index = i

				if action == 'deliver':
					available = next_states[0]
					unavailable = next_states[1]

					if node.pose_id in node.observations:
						last_observation_value = last_observation[0]
						last_observation_time = last_observation[1]
						a_priori_prob = self.avails[node.pose_id].get_prediction(self.states[available].time) 			# FIXME +1 minute to do delivery
						avail_prob = combine_probabilities(a_priori_prob, self.mu, self.states[available].time, last_observation_value, last_observation_time)
					else:
						avail_prob = self.avails[node.pose_id].get_prediction(self.states[available].time)

					expected_reward = avail_prob*self.expected_reward(available, maintenance_reward_collected) + (1.0 - avail_prob)*self.expected_reward(unavailable, maintenance_reward_collected)
					score = self.exploration_score(expected_reward, node.visits, num_visits)
					if score > best_score:
						best_score = score
						best_child_index = i

		return best_child_index


	def populate_children(self, node_id):

		node = self.nodes[node_id]

		### move actions
		for neighbor in self.spatial_graph.neighbors(node.pose_id):
			dist = self.spatial_graph[node.pose_id][neighbor]['weight']
			time = node.time + dist
			if time <= (self.start_time + self.budget):
				neighbor_node_id = self.id_form(neighbor, time, node.observations, node.requests_left_to_deliver)
				if not(neighbor_node_id in self.nodes):
					neighbor_node = MCTS_Node(neighbor_node_id, neighbor, time, node.observations, node.requests_left_to_deliver, node.requests_delivered)
					self.nodes[neighbor_node_id] = neighbor_node

				# add child
				node.children.append(['move', 0, [neighbor_node_id], [dist]])

		# self move
		neighbor = node.pose_id
		dist = 1
		time = node.time + dist
		if time <= (self.start_time + self.budget):
			neighbor_node_id = self.id_form(neighbor, time, node.observations, node.requests_left_to_deliver)
			if not(neighbor_node_id in self.nodes):
				neighbor_node = MCTS_Node(neighbor_node_id, neighbor, time, node.observations, node.requests_left_to_deliver, node.requests_delivered)
				self.nodes[neighbor_node_id] = neighbor_node

			# add child
			node.children.append(['move', 0, [neighbor_node_id], [dist]])


		### maintenance action
		if node.pose_id == self.maintenance_node:
			neighbor = node.pose_id
			dist = 1
			time = node.time + dist
			if time <= (self.start_time + self.budget):
				neighbor_node_id = self.id_form(neighbor, time, node.observations, node.requests_left_to_deliver)
				if not(neighbor_node_id in self.nodes):
					neighbor_node = MCTS_Node(neighbor_node_id, neighbor, time, node.observations, node.requests_left_to_deliver, node.requests_delivered)
					self.nodes[neighbor_node_id] = neighbor_node
				node.children.append(['maintenance', 0, [neighbor_node_id], [dist]])


		### observe action
		if node.pose_id in node.requests_left_to_deliver:
			neighbor = node.pose_id
			dist = 1
			time = node.time + dist
			if time <= (self.start_time + self.budget):
				next_states = []
				# available
				available_observations = copy.deepcopy(node.observations)
				available_observations[node.pose_id] = [1, node.time]
				available_node_id = self.id_form(neighbor, time, available_observations, node.requests_left_to_deliver)
				if not(available_node_id in self.nodes):
					available_node = MCTS_Node(available_node_id, neighbor, time, available_observations, node.requests_left_to_deliver, node.requests_delivered)
					self.nodes[available_node_id] = available_node
				next_states.append(available_node_id)
				# unavailable
				unavailable_observations = copy.deepcopy(node.observations)
				unavailable_observations[node.pose_id] = [0, node.time]
				unavailable_node_id = self.id_form(neighbor, time, unavailable_observations, node.requests_left_to_deliver)
				if not(unavailable_node_id in self.nodes):
					unavailable_node = MCTS_Node(unavailable_node_id, neighbor, time, unavailable_observations, node.requests_left_to_deliver, node.requests_delivered)
					self.nodes[unavailable_node_id] = unavailable_node
				next_states.append(unavailable_node_id)
				node.children.append(['observe', 0, next_states, [dist, dist]])


		### deliver action
		if node.pose_id in node.requests_left_to_deliver:
			dist = ucs(node.pose_id, self.distribution_node)
			success_time = node.time + dist*2
			failure_time = node.time + dist*3
			if success_time <= (self.start_time + self.budget):
				# available
				success_requests_left_to_deliver = node.requests_left_to_deliver
				success_requests_left_to_deliver.remove(node.pose_id)
				success_requests_delivered = node.requests_delivered
				success_requests_delivered.append(node.pose_id)
				success_node_id = self.id_form(node.pose_id, success_time, node.observations, success_requests_left_to_deliver, success_requests_delivered)
				if not(success_node_id in self.nodes):
					success_node = MCTS_Node(success_node_id, node.pose_id, success_time, node.observations, success_requests_left_to_deliver, success_requests_delivered)
					self.nodes[success_node_id] = success_node
				next_states.append(success_node_id)
				# unavailable
				failure_observations = copy.deepcopy(node.observations)
				failure_observations[node.pose_id] = [0, node.time + dist*2]
				if failure_time > (self.start_time + self.budget):
					failure_time = self.start_time + self.budget
				failure_node_id = self.id_form(self.distribution_node, failure_time, failure_observations, node.requests_left_to_deliver, node.requests_delivered)
				if not(failure_node_id in self.nodes):
					failure_node = MCTS_Node(failure_node_id, self.distribution_node, failure_time, failure_observations, node.requests_left_to_deliver, node.requests_delivered)
					self.nodes[failure_node_id] = failure_node
				next_states.append(failure_node_id)
				node.children.append(['deliver', 0, next_states, [dist*2, dist*3]])

		node.unexplored_indices = list(range(len(node.children)))
		self.nodes[node_id] = node
		return


	# def update_current_state():
	# 	#













	# def plan(self, planning_horizon, max_iterations):
	# 	for iteration in range(max_iterations):

	# 		if (iteration % 1000) == 0:
	# 			print ("iteration: " + str(iteration))

	# 		node, reward = self.expand(self.root_node, planning_horizon-1)
	# 		self.root_node = node

	
	# def expand(self, node, planning_horizon):
	# 	node.visits += 1
	# 	if planning_horizon > 0:

	# 		# if no children, add children
	# 		if len(node.children) == 0:
	# 			node = self.populate_children(node)

	# 		child_node_index = node.choose_child_for_exploration()
	# 		child_node, future_reward = self.expand(node.children[child_node_index], planning_horizon-1)
	# 		node.children[child_node_index] = child_node
	# 		node.cum_future_reward += future_reward
	# 		reward = node.node_reward + future_reward
	# 		return node, reward

	# 	else:
	# 		simulated_reward = self.simulate(node)
	# 		node.cum_future_reward += simulated_reward
	# 		reward = node.node_reward + simulated_reward
	# 		return node, reward


	# def simulate(self, node):
	# 	return 0.0

	# def get_plan(self, planning_horizon):
	# 	action_plan, pose_plan = self.expand_plan(self.root_node, planning_horizon)
	# 	# plan = [node.action_taken] + future_plan
	# 	return action_plan[1:], pose_plan[1:]

	# def expand_plan(self, node, planning_horizon):
	# 	if (planning_horizon > 0) and len(node.children) > 0:
	# 		child_node_index = node.get_best_child()
	# 		future_action_plan, future_pose_plan = self.expand_plan(node.children[child_node_index], planning_horizon)
	# 		action_plan = [node.action_taken] + future_action_plan
	# 		pose_plan = [node.state.robot_pose] + future_pose_plan
	# 		return action_plan, pose_plan
	# 	else:
	# 		return [node.action_taken], [node.state.robot_pose]







		


	# def make_choice(self):
	# 	best_children = []
	# 	most_visits = float('-inf')

	# 	for child in self.root_node.children:
	# 		if child.visits > most_visits:
	# 			most_visits = child.visits
	# 			best_children = [child]
	# 		elif child.visits == most_visits:
	# 			best_children.append(child)

	# 	return random.choice(best_children)

	# def make_exploratory_choice(self):
	# 	children_visits = map(lambda child: child.visits, self.root_node.children)
	# 	children_visit_probabilities = [visit / self.root_node.visits for visit in children_visits]
	# 	random_probability = random.uniform(0, 1)
	# 	probabilities_already_counted = 0.

	# 	for i, probability in enumerate(children_visit_probabilities):
	# 		if probabilities_already_counted + probability >= random_probability:
	# 			return self.root_node.children[i]

	# 		probabilities_already_counted += probability

	# def simulate(self, expansion_count = 1):
	# 	for i in range(expansion_count):
	# 		current_node = self.root_node

	# 		while current_node.expanded:
	# 			current_node = current_node.get_preferred_child(self.root_node)

	# 		self.expand(current_node)

	# def expand(self, node):
	# 	self.child_finder(node, self)

	# 	for child in node.children:
	# 		child_win_value = self.node_evaluator(child, self)

	# 		if child_win_value != None:
	# 			child.update_win_value(child_win_value)

	# 		if not child.is_scorable():
	# 			self.random_rollout(child)
	# 			child.children = []

	# 	if len(node.children):
	# 		node.expanded = True

	# def random_rollout(self, node):
	# 	self.child_finder(node, self)
	# 	child = random.choice(node.children)
	# 	node.children = []
	# 	node.add_child(child)
	# 	child_win_value = self.node_evaluator(child, self)

	# 	if child_win_value != None:
	# 		node.update_win_value(child_win_value)
	# 	else:
	# 		self.random_rollout(child)




















# class GraphNode:
# 	def __init__(self):
# 		self.neighbors = {}

# 	def get_neighbors(self):
# 		return self.neighbors.keys()

# class Graph:
# 	def __init__(self):
# 		self.vertices = {}
# 		self.rooms = []

# 	def initialize_hallway_graph(self, num_floors, num_rooms_per_floor, hallway_connectivity=2, maintenance=True, hallway_distance=1.0/12.0, floor_distance = 1.0):
# 		floor_entryways = []
# 		self.rooms = []
# 		for floor in range(num_floors):
# 			entryway = 'entryway_' + str(floor)
# 			self.vertices[entryway] = GraphNode()
# 			floor_entryways.append(entryway)
# 			prev_hallway = entryway

# 			for junction in range(int(num_rooms_per_floor/hallway_connectivity)):
# 				hallway = 'hallway_' + str(floor) + '_' + str(junction)
# 				self.vertices[hallway] = GraphNode()
# 				self.add_edge(hallway, prev_hallway, hallway_distance)

# 				for room_number in range(hallway_connectivity):
# 					room = 'room_' + str(floor) + '_' + str(junction) + '_' + str(room_number)
# 					self.rooms.append(room)
# 					self.vertices[room] = GraphNode()
# 					self.add_edge(room, hallway, hallway_distance)
# 				prev_hallway = hallway

# 		for i in range(len(floor_entryways)-1):
# 			self.add_edge(floor_entryways[i], floor_entryways[i+1], floor_distance)

# 		if maintenance:
# 			self.maintenance_node = 'maintenance_node'
# 			self.vertices[self.maintenance_node] = GraphNode()
# 			self.add_edge(self.maintenance_node, floor_entryways[0], floor_distance)

# 	def initialize_fully_connected_graph(self, num_floors, num_rooms_per_floor, hallway_connectivity=1, maintenance=True, hallway_distance=1.0/12.0, floor_distance = 1.0):
# 		self.rooms = []
# 		prev_landing = None
# 		if maintenance: 
# 			self.maintenance_node = 'maintenance_node'
# 			self.vertices[self.maintenance_node] = GraphNode()
# 			prev_landing = self.maintenance_node

# 		for floor in range(num_floors):
# 			floor_rooms = []
# 			prev_room = None
# 			for room_number in range(num_rooms_per_floor):
# 				room = 'room_' + str(floor) + '_' + str(room_number)
# 				self.rooms.append(room)
# 				floor_rooms.append(room)
# 				self.vertices[room] = GraphNode()

# 				if room_number == 0:
# 					self.add_edge(room, prev_landing, floor_distance)
# 				else:
# 					self.add_edge(room, prev_room, hallway_distance)
# 				prev_room = room
# 		self.fully_connect()

# 	def fully_connect(self):
# 		for vertex_1 in self.vertices.keys():
# 			for vertex_2 in self.vertices.keys():
# 				if vertex_1 != vertex_2:
# 					if vertex_2 not in self.vertices[vertex_1].get_neighbors():
# 						min_dist = self.ucs(vertex_1, vertex_2)
# 						self.add_edge(vertex_1, vertex_2, min_dist)

# 	def ucs(self, start, end):
# 		closed_list = []
# 		h = []

# 		# enqueue
# 		closed_list.append(start)
# 		neighbors = self.vertices[start].get_neighbors()
# 		for neighbor in neighbors:
# 			dist = self.get_distance(start, neighbor)
# 			heapq.heappush(h, (dist, neighbor))

# 		while len(h) != 0:
# 			top = heapq.heappop(h)
# 			if top[1] == end:
# 				return top[0]
# 			else:
# 				# enqueue
# 				closed_list.append(top[1])
# 				neighbors = self.vertices[top[1]].get_neighbors()
# 				for neighbor in neighbors:
# 					if neighbor not in closed_list:
# 						dist = self.get_distance(top[1], neighbor)
# 						heapq.heappush(h, (top[0] + dist, neighbor))

# 		print ("COULD NOT CONNECT: " + str(start) + ' ' + str(end))
# 		return float("inf")


# 	def add_edge(self, v1, v2, dist):
# 		if v1 not in self.vertices:
# 			self.vertices[v1] = GraphNode()
# 		if v2 not in self.vertices:
# 			self.vertices[v2] = GraphNode()

# 		self.vertices[v1].neighbors[v2] = dist
# 		self.vertices[v2].neighbors[v1] = dist

# 	def get_distance(self, v1, v2):
# 		neighbors = self.vertices[v1].neighbors
# 		dist = neighbors[v2]
# 		return dist

# 	def is_maintenance_node(self, node):
# 		return self.maintenance_node == node


# class World:

# 	def __init__(self, initial_time, max_time, num_floors, num_rooms_per_floor, initial_pose, planning_horizon):
# 		self.curr_time = initial_time
# 		self.max_time = max_time
# 		self.graph = Graph()
# 		# self.graph.initialize_hallway_graph(num_floors, num_rooms_per_floor)
# 		self.graph.initialize_fully_connected_graph(num_floors, num_rooms_per_floor)

# 		self.robot_pose = initial_pose
# 		self.generate_deterministic_room_occupancy_schedules()
# 		# self.current_tasks = self.sample_tasks(initial_time, planning_horizon)
# 		self.current_tasks = {}
# 		self.generate_deterministic_tasks()
# 		self.reward = 0.0

# 		self.deliver_time = 1.0
# 		self.observe_time = .5
# 		self.maintenance_time = 10.0
# 		self.delivery_reward = 100000
# 		self.maintenance_reward = .01
# 		self.bad_action_penalty = -10


# 	def generate_deterministic_room_occupancy_schedules(self):
# 		self.schedules = {}

# 		# # room 0 always occupied during working hours
# 		# self.schedules[self.graph.rooms[0]] = lambda x: int(8 <= ((x/60.0) % 24) <= 17)

# 		# # room 1 occupied before noon
# 		# self.schedules[self.graph.rooms[1]] = lambda x: int(8 <= ((x/60.0) % 24) <= 12)

# 		# # room 2 occupied during even working hours
# 		# self.schedules[self.graph.rooms[2]] = lambda x: int((8 <= ((x/60.0) % 24) <= 17) and ((int(math.floor((x/60) % 24))%2) == 0))

# 		# # room 3 occupied during odd working hours
# 		# self.schedules[self.graph.rooms[3]] = lambda x: int((8 <= ((x/60.0) % 24) <= 17) and ((int(math.floor((x/60) % 24))%2) == 1))




# 		# all available all the time
# 		self.schedules[self.graph.rooms[0]] = lambda x: int(8 <= ((x/60.0) % 24) <= 17)
# 		self.schedules[self.graph.rooms[1]] = lambda x: int(8 <= ((x/60.0) % 24) <= 17)
# 		self.schedules[self.graph.rooms[2]] = lambda x: int(8 <= ((x/60.0) % 24) <= 17)
# 		self.schedules[self.graph.rooms[3]] = lambda x: int(8 <= ((x/60.0) % 24) <= 17)
		



# 	def generate_deterministic_tasks(self):
# 		for room in self.graph.rooms:
# 			self.current_tasks[room] = [self.curr_time]
# 		# print (self.current_tasks.keys())
# 			# if room in self.current_tasks.keys():
# 			# 	self.current_tasks[room].append(self.curr_time)
# 			# else:
# 			# 	self.current_tasks[room] = [self.curr_time]

# 	# def sample_tasks(self):
# 	# 	#

# 	def is_occupied(self, pose, time):
# 		# return bool(self.schedules[pose](time))
# 		occ = self.schedules[pose](time) == 1
# 		return occ

# 	def update(self, action, node):

# 		# print (action)
# 		# print ()

# 		if action == 'move':
# 			distance = self.graph.get_distance(self.robot_pose, node)
# 			if (self.curr_time + distance) <= self.max_time:
# 				self.curr_time += distance
# 				self.robot_pose = node
# 				return 1

# 			else:
# 				print ("END OF LIFE")
# 				return 0
# 			# 	# END OF LINE

# 		# if action == 'observe':
# 		# 	#

# 		if action == 'deliver':
# 			if (self.curr_time + self.deliver_time) <= self.max_time:

# 				# successful delivery
# 				if self.is_occupied(self.robot_pose, self.curr_time) and (self.robot_pose in self.current_tasks.keys()):
# 					self.reward += len(self.current_tasks[self.robot_pose])*self.delivery_reward
# 					# self.remove_tasks(self.robot_pose)
# 					del self.current_tasks[self.robot_pose]

# 					print ("Delivered: " + str(self.robot_pose) + ", " + str(self.curr_time/60.0))
# 					self.curr_time += self.deliver_time
# 					return 1

# 					# observation

# 				else:
# 					print ("Failed delivery: " + str(self.robot_pose) + ", " + str(self.curr_time/60.0))
# 					self.reward += self.bad_action_penalty
# 					self.curr_time += self.deliver_time
# 					return 1

# 					# observation
# 			else:
# 				print ("END OF LIFE")
# 				return 0
# 			# 	# END OF LINE

# 		if action == 'maintenance':
# 			if (self.curr_time + self.maintenance_time) <= self.max_time:
# 				if self.graph.is_maintenance_node(self.robot_pose):
# 					print ("Maintenance: " + str(self.robot_pose) + ", " + str(self.curr_time/60.0))
# 					self.reward += self.maintenance_reward
# 					self.curr_time += self.maintenance_time
# 					return 1

# 				else:
# 					print ("Failed maintenance: " + str(self.robot_pose) + ", " + str(self.curr_time/60.0))
# 					self.reward += self.bad_action_penalty
# 					self.curr_time += self.maintenance_time
# 					return 1

# 			else:
# 				print ("END OF LIFE")
# 				return 0
# 			# 	# END OF LINE

# 	# def visualize():
# 	# 	#



# def main():
# 	initial_time = 8.00*60
# 	max_time = 8.00*60 + 2.0 * 60
# 	num_floors = 2
# 	num_rooms_per_floor = 2
# 	# initial_pose = 'hallway_0_0'
# 	initial_pose = 'room_0_0'
# 	# planning_horizon = 2.0 * 60
# 	planning_horizon = 100
# 	max_iterations = 10000
# 	world = World(initial_time, max_time, num_floors, num_rooms_per_floor, initial_pose, planning_horizon)

# 	cont = 1
# 	while (world.curr_time < world.max_time) and (cont == 1):
# 		start_node  = Node(world.robot_pose, world.curr_time, world.current_tasks, None, 0.0)
# 		planner = MCTS(start_node, world)
# 		planner.plan(planning_horizon, max_iterations)

# 		action_plan, pose_plan  = planner.get_plan(planning_horizon)

# 		# print (action_plan)
# 		# print (pose_plan)
# 		# print ()

# 		# action plan, not policy
# 		for i in range(len(action_plan)):
# 			action = action_plan[i]
# 			pose = pose_plan[i]
# 			cont = world.update(action, pose)
# 			if cont == 0:
# 				break

# 	print ("Total reward: " + str(world.reward))

# 	# print (action_plan)

# 	# while world.curr_time < world.max_time:

# 	# 	# IF REPLAIN
# 	# 		# PLAN
# 	# 		planner.plan(planning_horizon, max_iterations)

# 	# 	# CURR ACTION = GET ACTION (PLAN)
# 	# 	world.update(action)



# if __name__ == "__main__":
#     main()