import numpy as np
import random
import copy
import math
import heapq
import networkx as nx  



class GraphNode:
    def __init__(self):
        self.neighbors = {}

    def get_neighbors(self):
        return self.neighbors.keys()

### Supporting Graph class to represent 2D graph
class Graph:
    def __init__(self):
        self.vertices = {}
        self.rooms = []

    def read_graph_from_file(self, input_filename):
        # read known connections
        lines = [line.rstrip('\n') for line in open(input_filename)]
        for line in lines:
            line = line.split()
            node_a = line[0]
            node_b = line[1]
            dist = float(line[2])
            dist = max(int(round(float(dist)/60)), 1)           # convert seconds to minutes
            self.add_edge(node_a, node_b, dist)

    def create_cost_matrix_from_graph_willow(self, output_filename):
        # calculate all pairwise connections
        self.fully_connect()

        print ("connections done")

        # fill in cost matrix
        node_ids = ["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10","R11","R12","R13","R14","R15","R16","R17","R18","R19","R20",
                    "R21","R22","R23","R24","R25","R26","R27","R28","R29","R30","R31","R32","R33","R34","R35","R36","R37","R38","R39","R40",
                    "R41","R42","R43","C1","C2","C3","C4","C5","C6","C7","C8","C9","H1","H2","H3","H4","H5","H6","H7","H8","H9","H10","H11","H12","H13","H14"]
        num_nodes = len(node_ids)
        cost_matrix = np.zeros((num_nodes,num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    # cost_matrix[i][j] = 111111
                    cost_matrix[i][j] = 1
                else:
                    cost_matrix[i][j] = self.get_distance(node_ids[i], node_ids[j])

        # write cost matrix
        f = open(output_filename, 'w')
        for i in range(num_nodes):
            for j in range(num_nodes):
                f.write(str(cost_matrix[i][j]) + " ")
            f.write("\n")
        f.close()

    def initialize_hallway_graph(self, num_floors, num_rooms_per_floor, hallway_connectivity=2, maintenance=True, hallway_distance=1.0/12.0, floor_distance = 1.0):
        floor_entryways = []
        self.rooms = []
        for floor in range(num_floors):
            entryway = 'entryway_' + str(floor)
            self.vertices[entryway] = GraphNode()
            floor_entryways.append(entryway)
            prev_hallway = entryway

            for junction in range(int(num_rooms_per_floor/hallway_connectivity)):
                hallway = 'hallway_' + str(floor) + '_' + str(junction)
                self.vertices[hallway] = GraphNode()
                self.add_edge(hallway, prev_hallway, hallway_distance)

                for room_number in range(hallway_connectivity):
                    room = 'room_' + str(floor) + '_' + str(junction) + '_' + str(room_number)
                    self.rooms.append(room)
                    self.vertices[room] = GraphNode()
                    self.add_edge(room, hallway, hallway_distance)
                prev_hallway = hallway

        for i in range(len(floor_entryways)-1):
            self.add_edge(floor_entryways[i], floor_entryways[i+1], floor_distance)

        if maintenance:
            self.maintenance_node = 'maintenance_node'
            self.vertices[self.maintenance_node] = GraphNode()
            self.add_edge(self.maintenance_node, floor_entryways[0], floor_distance)

    def initialize_fully_connected_graph(self, num_floors, num_rooms_per_floor, hallway_connectivity=1, maintenance=True, hallway_distance=1.0/12.0, floor_distance = 1.0):
        self.rooms = []
        prev_landing = None
        if maintenance: 
            self.maintenance_node = 'maintenance_node'
            self.vertices[self.maintenance_node] = GraphNode()
            prev_landing = self.maintenance_node

        for floor in range(num_floors):
            floor_rooms = []
            prev_room = None
            for room_number in range(num_rooms_per_floor):
                room = 'room_' + str(floor) + '_' + str(room_number)
                self.rooms.append(room)
                floor_rooms.append(room)
                self.vertices[room] = GraphNode()

                if room_number == 0:
                    self.add_edge(room, prev_landing, floor_distance)
                else:
                    self.add_edge(room, prev_room, hallway_distance)
                prev_room = room
        self.fully_connect()

    def fully_connect(self):
        count = 1
        for vertex_1 in self.vertices.keys():
            for vertex_2 in self.vertices.keys():
                if vertex_1 != vertex_2:
                    if vertex_2 not in self.vertices[vertex_1].get_neighbors():
                        min_dist = self.ucs(vertex_1, vertex_2)
                        self.add_edge(vertex_1, vertex_2, min_dist)
            print (count)
            count += 1

    def ucs(self, start, end):
        closed_list = []
        h = []

        # enqueue
        closed_list.append(start)
        neighbors = self.vertices[start].get_neighbors()
        for neighbor in neighbors:
            dist = self.get_distance(start, neighbor)
            heapq.heappush(h, (dist, neighbor))

        while len(h) != 0:
            top = heapq.heappop(h)
            if top[1] == end:
                return top[0]
            else:
                # enqueue
                closed_list.append(top[1])
                neighbors = self.vertices[top[1]].get_neighbors()
                for neighbor in neighbors:
                    if neighbor not in closed_list:
                        dist = self.get_distance(top[1], neighbor)
                        heapq.heappush(h, (top[0] + dist, neighbor))

        print ("COULD NOT CONNECT: " + str(start) + ' ' + str(end))
        return float("inf")


    def add_edge(self, v1, v2, dist):
        if v1 not in self.vertices:
            self.vertices[v1] = GraphNode()
        if v2 not in self.vertices:
            self.vertices[v2] = GraphNode()

        self.vertices[v1].neighbors[v2] = dist
        self.vertices[v2].neighbors[v1] = dist

    def get_distance(self, v1, v2):
        neighbors = self.vertices[v1].neighbors
        dist = neighbors[v2]
        return dist



class STGraphNode:
    def __init__(self):
        self.id = None
        self.t = 0.0
        self.name = None
        self.prob = 0.0
        self.profit = 0.0
        self.weight = 0.0
        self.sum = -float("inf")
        self.parent = -1
        self.successors = []
        self.indegree = 0
        self.path = []
        self.serviced_probs = {}


### Modification of representation proposed in Ma, Zhibei, et al. "A Spatio-Temporal Representation for the Orienteering Problem with Time-Varying Profits." IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2017
class SpatioTemporalGraph:
    def __init__(self, availability_models, model_variances, mu, num_intervals, budget, time_interval, maintenance_node, maintenance_reward, deliver_reward):
        self.vertices = {}
        self.start_node = None
        self.availability_models = availability_models
        self.model_variances = model_variances
        self.mu = mu
        self.num_intervals = num_intervals
        self.budget = budget
        self.time_interval = time_interval
        self.maintenance_node = maintenance_node
        self.maintenance_reward = maintenance_reward
        self.deliver_reward = deliver_reward

    ### Create STGraph, node for each spatial node/time slice. Edges connect nodes at different time slices according to traversal costs.
    def build_graph(self, spatial_graph, graph_start_node_id, graph_start_time, requests_left_to_deliver, observations, incorporate_observation, incorporate_observation_hack, variance_bias):
        graph_start_node = STGraphNode()
        graph_start_node.id = graph_start_node_id
        graph_start_node.t = graph_start_time
        graph_start_node.name = graph_start_node_id + "_" + str(graph_start_time)
        graph_start_node.sum = 0.0
        if graph_start_node.id == self.maintenance_node:
            graph_start_node.profit = self.maintenance_reward
            graph_start_node.weight = self.maintenance_reward
            graph_start_node.sum = self.maintenance_reward
        self.vertices[graph_start_node.name] = graph_start_node
        self.start_node = graph_start_node.name

        # for v in spatial_graph.vertices.keys():
        for v in spatial_graph:
            for t in range(self.num_intervals):
                node_time = self.vertices[self.start_node].t + (t*self.time_interval)
                node_name = v + "_" + str(node_time)
                if node_name in self.vertices:
                    st_node = self.vertices[node_name]
                else:
                    st_node = STGraphNode()
                    st_node.id = v
                    st_node.t = node_time
                    st_node.name = v + "_" + str(st_node.t)
                if v in requests_left_to_deliver:
                    if incorporate_observation:
                        if incorporate_observation_hack:
                            if st_node.id in observations.keys():
                                last_observation = observations[v][0]
                                last_observation_time = observations[v][1]
                                st_node.prob = self.combine_probabilities_hack(v, st_node.t, last_observation, last_observation_time)
                                st_node.profit = bernoulli_variance_biasing(st_node.prob, variance_bias, self.deliver_reward)       # should be updated to handle more than bernoulli variance
                                st_node.serviced_probs[st_node.id] = st_node.prob
                            else:
                                st_node.prob = self.availability_models[v][int(st_node.t/self.time_interval)]
                                st_node.profit = bernoulli_variance_biasing(st_node.prob, variance_bias, self.deliver_reward)
                                st_node.serviced_probs[st_node.id] = st_node.prob
                        else:
                            if st_node.id in observations.keys():
                                last_observation = observations[v][0]
                                last_observation_time = observations[v][1]
                                st_node.prob = self.combine_probabilities(v, st_node.t, last_observation, last_observation_time)
                                st_node.profit = bernoulli_variance_biasing(st_node.prob, variance_bias, self.deliver_reward)
                                st_node.serviced_probs[st_node.id] = st_node.prob
                            else:
                                st_node.prob = self.availability_models[v][int(st_node.t/self.time_interval)]
                                st_node.profit = bernoulli_variance_biasing(st_node.prob, variance_bias, self.deliver_reward)
                                st_node.serviced_probs[st_node.id] = st_node.prob
                    else:
                        st_node.prob = self.availability_models[v][int(st_node.t/self.time_interval)]
                        st_node.profit = bernoulli_variance_biasing(st_node.prob, variance_bias, self.deliver_reward)
                        st_node.serviced_probs[st_node.id] = st_node.prob
                elif v == self.maintenance_node:
                    st_node.profit = self.maintenance_reward
                else:
                    st_node.profit = 0.0
                st_node.weight = st_node.profit

                # for each neighbor
                # neighbors = spatial_graph.vertices[v].get_neighbors()
                # for neighbor in neighbors:
                for neighbor in spatial_graph.neighbors(v):
                    # dist = spatial_graph.get_distance(v, neighbor)
                    dist = spatial_graph[v][neighbor]['weight']

                    # if travel cost doesnt exceed budget add neighbor to dag and increase its indegree
                    if (st_node.t + dist) <= (graph_start_time + self.budget):
                        neighbor_name = neighbor + "_" + str(st_node.t + dist)
                        if neighbor_name in self.vertices:
                            neighbor_node = self.vertices[neighbor_name]
                        else:
                            neighbor_node = STGraphNode()
                            neighbor_node.id = neighbor
                            neighbor_node.t = st_node.t + dist
                            neighbor_node.name = neighbor_name
                        neighbor_node.indegree += 1
                        self.vertices[neighbor_name] = neighbor_node

                        st_node.successors.append(neighbor_name)
                # add self vertex
                # if v not in requests_left_to_deliver:
                dist = 1
                if (st_node.t + dist) <= (graph_start_time + self.budget):
                    neighbor_name = v + "_" + str(st_node.t + dist)
                    if neighbor_name in self.vertices:
                        neighbor_node = self.vertices[neighbor_name]
                    else:
                        neighbor_node = STGraphNode()
                        neighbor_node.id = v
                        neighbor_node.t = st_node.t + dist
                        neighbor_node.name = neighbor_name
                    neighbor_node.indegree += 1
                    self.vertices[neighbor_name] = neighbor_node

                    st_node.successors.append(neighbor_name)

                # # add self vertex for start node
                # if ((v + "_" + str(t)) == self.start_node) and (v in requests_left_to_deliver):

                self.vertices[node_name] = st_node


    ### Topologically sort nodes to allow for efficient DP shortest path (max profit path) calculations
    def topological_sort(self):
        self_copy = copy.deepcopy(self)
        L = []
        S = set()
        for node_name in self.vertices.keys():
            node = self.vertices[node_name]
            if node.indegree == 0:
                S.add(node_name)
                node.path = [node.id]
                self.vertices[node_name] = node

        # while there is at least one node without incoming edges
        while len(S) != 0:
            node_name = S.pop()
            L.append(node_name)
            node = self_copy.vertices[node_name]
            for successor_name in node.successors:
                successor = self_copy.vertices[successor_name]
                successor.indegree -= 1
                if successor.indegree == 0:
                    S.add(successor_name)
                self_copy.vertices[successor_name] = successor
        return L

    ### DP based calculation of max profit path from starting node within budget
    def calc_max_profit_path(self, L, node_requests, multiple_visits):

        # for each node in topological order
        for node_name in L:
            node = self.vertices[node_name]

            # for each successor
            successors = node.successors

            for successor_name in successors:
                successor = self.vertices[successor_name]

                # if successor is a delivery node and has not been already visited up to that point
                if successor.id in node_requests: 
                    if (successor.id not in node.path[1:]):

                        # if going through node is the best way to get to successor, update successors parent
                        if (node.sum + successor.profit) > successor.sum:
                            # print ("successor sum: " + str(successor_sum))
                            successor.sum = node.sum + successor.profit
                            # print ("successor sum: " + str(successor_sum))
                            successor.parent = node_name
                            successor.path = node.path + [successor.id]
                            successor.serviced_probs = copy.deepcopy(node.serviced_probs)
                            successor.serviced_probs[successor.id] = successor.prob
                            # print ("new path: " + successor.path)
                            # print()
                            self.vertices[successor_name] = successor

                    else:
                        if multiple_visits:
                            not_visited = 1.0 - node.serviced_probs[successor.id]
                            successor_profit = not_visited*successor.profit
                                
                            if (node.sum + successor_profit) > successor.sum:
                                successor.sum = node.sum + successor_profit
                                successor.parent = node_name
                                successor.path = node.path + [successor.id]
                                successor.serviced_probs = copy.deepcopy(node.serviced_probs)
                                successor.serviced_probs[successor.id] = node.serviced_probs[successor.id] + not_visited*successor.prob

                                self.vertices[successor_name] = successor

                        else:
                            if (node.sum + successor.profit) > successor.sum:
                                successor.sum = node.sum + 0.0
                                successor.parent = node_name
                                successor.path = node.path + [successor.id]
                                self.vertices[successor_name] = successor

                else:
                    # if going through node is the best way to get to successor, update successors parent
                    successor_profit = 0.0
                    if successor.id == self.maintenance_node:
                        successor_profit = self.maintenance_reward
                    if (node.sum + successor_profit) > successor.sum:
                        successor.sum = node.sum + successor_profit
                        successor.parent = node_name
                        successor.path = node.path + [successor.id]
                        successor.serviced_probs = copy.deepcopy(node.serviced_probs)
                        self.vertices[successor_name] = successor

        max_sum = -float("inf")
        end_node = None
        for node_name in self.vertices.keys():
            node = self.vertices[node_name]
            if node.sum > max_sum:
                max_sum = node.sum
                end_node = node_name

        # backtrack from end node to get path
        path = self.vertices[end_node].path
        return path


    ### Bayesian update of model availability probabilities with info from latest observation (respecting temporal persistence)
    def combine_probabilities(self, node_id, curr_time, last_observation, last_observation_time):
        a_priori_prob = self.availability_models[node_id][int(curr_time/self.time_interval)]
        likelihood = persistence_prob(self.mu, curr_time-last_observation_time, last_observation)
        # if last_observation == 1:
        #     evidence_prob = availability_model(last_observation_time)
        # else:
        #     evidence_prob = 1 - availability_model(last_observation_time)
        evidence_prob = (likelihood*a_priori_prob) + (1.0-likelihood)*(1.0-a_priori_prob)
        if evidence_prob < .001:
            new_prob = .99
        else:
            new_prob = likelihood*a_priori_prob/evidence_prob         # Bayesian update of last observation times occ prior
        return new_prob

    ### Simplistic method for accounting for observations. Zeroes out probability for fixed amount following negative observation
    def combine_probabilities_hack(self, node_id, curr_time, last_observation, last_observation_time):
        a_priori_prob = self.availability_models[node_id][int(curr_time/self.time_interval)]
        new_prob = a_priori_prob
        if last_observation == 0:
            if curr_time < (last_observation_time + (self.mu/2)):
                new_prob = 0.0
        return new_prob


### Add random noise to availability model
def add_random_noise(availability_model, noise_amplitude, availability_chance):
    f = lambda x: min(max(availability_model(x) + (random.random() -.5)*(noise_amplitude*2), 1.0 - availability_chance), availability_chance)
    return f

### Temporal persistence per Toris, Russell, and Sonia Chernova. "Temporal Persistence Modeling for Object Search." IEEE International Conference on Robotics and Automation (ICRA). 2017.
def persistence_prob(mu, delta_t, last_observation):
    if last_observation == 1:
        return math.exp(-(1.0/mu)*(delta_t))
    else:
        return 1.0 - math.exp(-(1.0/mu)*(delta_t))

def bernoulli_variance(availability_prob):
    return availability_prob*(1-availability_prob)

### Sample model availability parameter assuming Bernoulli variance
def sample_bernoulli_avialability_model(availability_model):
    avails = []
    for avail in availability_model:
        new_avail = max(min(random.gauss(avail, math.sqrt(bernoulli_variance(avail))), 0.99), 0.01)
        avails.append(new_avail)
    return avails

### Artificially increase expected reward from reliable (low variance) nodes
def bernoulli_variance_biasing(prob, variance_bias, deliver_reward):
    reward = deliver_reward*prob - variance_bias*bernoulli_variance(prob)
    assert(reward >= 0)
    return reward