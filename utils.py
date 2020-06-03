import yaml
import networkx as nx  
import copy
import math
from io import BytesIO
from PIL import Image
import heapq

def load_params(world_config_file, schedule_config_file, planner_config_file, model_config_file):
    params = {}
    with open(world_config_file) as f:
        world_params = yaml.load(f, Loader=yaml.FullLoader)
    params['rooms'] = world_params['rooms']
    params['start_node_id'] = world_params['start_node_id']
    params['maintenance_node'] = world_params['maintenance_node']
    params['max_rooms'] = int(world_params['max_rooms'])
    params['max_traversal_cost'] = int(world_params['max_traversal_cost'])
    params['graph_generator_type'] = world_params['graph_generator_type']
    params['distance_scaling'] = world_params['distance_scaling']
    if 'graph_filename' in world_params.keys():
        params['graph_filename'] = world_params['graph_filename']

    with open(schedule_config_file) as f:
        schedule_params = yaml.load(f, Loader=yaml.FullLoader)
    params['budget'] = int(schedule_params['budget'])
    params['time_interval'] = int(schedule_params['time_interval'])
    params['start_time'] = int(schedule_params['start_time'])
    params['node_closeness'] = schedule_params['node_closeness']
    params['availabilities'] = schedule_params['availabilities']
    params['sampling_method'] = schedule_params['sampling_method']
    params['availability_length'] = int(schedule_params['availability_length'])
    params['availability_chance'] = float(schedule_params['availability_chance'])
    params['temporal_consistency'] = bool(int(schedule_params['temporal_consistency']))
    params['maintenance_reward'] = float(schedule_params['maintenance_reward'])
    params['deliver_reward'] = float(schedule_params['deliver_reward'])
    params['max_noise_amplitude'] = float(schedule_params['max_noise_amplitude'])
    params['schedule_generation_method'] = schedule_params['schedule_generation_method']
    params['num_intervals'] = int(params['budget']/params['time_interval'])

    with open(planner_config_file) as f:
        planner_params = yaml.load(f, Loader=yaml.FullLoader)
    params['variance_bias'] = float(planner_params['variance_bias'])
    params['num_paths'] = int(planner_params['num_paths'])
    params['num_worlds'] = int(planner_params['num_worlds'])

    with open(model_config_file) as f:
        model_params = yaml.load(f, Loader=yaml.FullLoader)
    params['spacing'] = int(model_params['spacing'])
    params['noise_scaling'] = float(model_params['noise_scaling'])
    params['use_gp'] = bool(int(model_params['use_gp']))

    return params


### Temporal persistence per Toris, Russell, and Sonia Chernova. "Temporal Persistence Modeling for Object Search." IEEE International Conference on Robotics and Automation (ICRA). 2017.
def persistence_prob(mu, delta_t, last_observation):
    if last_observation == 1:
        return .5 + .5*(math.exp(-(1.0/mu)*(delta_t)))
    else:
        return .5 - .5*(math.exp(-(1.0/mu)*(delta_t)))


### Bayesian update of model availability probabilities with info from latest observation (respecting temporal persistence)
def combine_probabilities(a_priori_prob, mu, curr_time, last_observation, last_observation_time):
    likelihood = persistence_prob(mu, curr_time-last_observation_time, last_observation)
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


def ucs(g, start, end):
    closed_list = []
    h = []

    # enqueue
    closed_list.append(start)
    # neighbors = g.vertices[start].get_neighbors()
    neighbors = g.neighbors(start)
    for neighbor in neighbors:
        # dist = g.get_distance(start, neighbor)
        dist = g[start][neighbor]['weight']
        heapq.heappush(h, (dist, neighbor))

    while len(h) != 0:
        top = heapq.heappop(h)
        if top[1] == end:
            return top[0]
        else:
            # enqueue
            closed_list.append(top[1])
            # neighbors = g.vertices[top[1]].get_neighbors()
            neighbors = g.neighbors(top[1])
            for neighbor in neighbors:
                if neighbor not in closed_list:
                    # dist = g.get_distance(top[1], neighbor)
                    dist = g[top[1]][neighbor]['weight']
                    heapq.heappush(h, (top[0] + dist, neighbor))

    print ("COULD NOT CONNECT: " + str(start) + ' ' + str(end))
    return float("inf")



### Visualize servicing execution over graph at given time slice
def visualize_graph(g, base_availability_models, true_schedule, availability_observations, curr_time_index, curr_node, node_requests, nodes_delivered, curr_time, mu, strategy, use_gp):

    if strategy == 'no_temp':
        incorporate_observation = False
    elif strategy == 'no_replan':
        incorporate_observation = False
    elif strategy == 'replan_no_observe':
        incorporate_observation = False
    elif strategy == 'hack_observe':
        incorporate_observation = True
    elif strategy == 'observe':
        incorporate_observation = True
    elif strategy == 'observe_mult_visits':
        incorporate_observation = True
    elif strategy == 'observe_sampling':
        incorporate_observation = True
    elif strategy == 'observe_sampling_variance_bias':
        incorporate_observation = True
    elif strategy == 'observe_sampling_mult_visits':
        incorporate_observation = True

    # availability_viz = 'schedule'
    availability_viz = 'prob'


    viz_g = copy.deepcopy(g)
    for v in viz_g:
        

        if not(v in node_requests):
            viz_g.nodes[v]['fillcolor'] = "gray"
        elif v in nodes_delivered:
            viz_g.nodes[v]['fillcolor'] = "greenyellow"
        else:
            if incorporate_observation:
                if v in availability_observations.keys():
                    if use_gp:
                        prob = combine_probabilities(base_availability_models[v].get_prediction(curr_time), mu, curr_time, availability_observations[v][0], availability_observations[v][1])
                    else:
                        prob = combine_probabilities(base_availability_models[v][curr_time_index], mu, curr_time, availability_observations[v][0], availability_observations[v][1])
                else:
                    if use_gp:
                        prob = base_availability_models[v].get_prediction(curr_time)
                    else:
                        prob = base_availability_models[v][curr_time_index]
            else:
                if use_gp:
                    prob = base_availability_models[v].get_prediction(curr_time)
                else:
                    prob = base_availability_models[v][curr_time_index]
            viz_g.nodes[v]['prob'] = prob
            viz_g.nodes[v]['schedule'] = true_schedule[v][curr_time_index]

            if availability_viz == 'schedule':
                # unavailable
                if viz_g.nodes[v]['schedule'] == 0:
                    viz_g.nodes[v]['fillcolor'] = "/blues9/1"
                
                # available
                else:
                    viz_g.nodes[v]['fillcolor'] = "/blues9/9"

            else:
                color_prob = int(round(viz_g.nodes[v]['prob']*10))
                if color_prob == 0:
                    viz_g.nodes[v]['fillcolor'] = "white"
                elif color_prob == 10:
                    viz_g.nodes[v]['fillcolor'] = "/blues9/9"
                else:
                    viz_g.nodes[v]['fillcolor'] = "/blues9/" + str(color_prob)


        if v == curr_node:
            viz_g.nodes[v]['label'] = v
            viz_g.nodes[v]['style'] = "filled, bold"
            viz_g.nodes[v]['fontcolor'] = "blue"
            viz_g.nodes[v]['color'] = "blue"
        else:
            viz_g.nodes[v]['label'] = v
            viz_g.nodes[v]['style'] = "filled"
            viz_g.nodes[v]['fontcolor'] = "black"
            viz_g.nodes[v]['color'] = "black"   

    pydot_graph = nx.nx_pydot.to_pydot(viz_g)
    png_str = pydot_graph.create_png(prog='dot')
    im_io = BytesIO()
    im_io.write(png_str)
    im_io.seek(0)
    img = Image.open(im_io) 

    return img



def read_graph_from_file(input_filename):
    g = nx.Graph()

    # read known connections
    lines = [line.rstrip('\n') for line in open(input_filename)]
    for line in lines:
        line = line.split()
        node_a = line[0]
        node_b = line[1]
        dist = float(line[2])
        dist = max(int(round(float(dist)/60)), 1)           # convert seconds to minutes

        # add nodes
        if not(g.has_node(node_a)):
            g.add_node(node_a)
        if not(g.has_node(node_b)):
            g.add_node(node_b)

        # add edges
        g.add_edge(node_a, node_b)
        g.add_edge(node_b, node_a)
        g[node_a][node_b]['weight'] = dist
        g[node_b][node_a]['weight'] = dist

    return g



def read_cost_matrix_from_file(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    cost_matrix = []
    for line in lines:
        row = []
        for item in line.split():
            # val = max(float(item)/60, 1)        # convert seconds to minutes
            row.append(float(item))
        cost_matrix.append(row)
    cost_matrix = np.array(cost_matrix)
    return cost_matrix