import yaml
import networkx as nx    

def load_params(world_config_file, schedule_config_file, planner_config_file):
    params = {}
    with open(world_config_file) as f:
        world_params = yaml.load(f, Loader=yaml.FullLoader)
    params['rooms'] = world_params['rooms']
    params['start_node_id'] = world_params['start_node_id']
    params['maintenance_node'] = world_params['maintenance_node']
    params['max_rooms'] = int(world_params['max_rooms'])
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
    params['num_intervals'] = int(params['budget']/params['time_interval'])

    with open(planner_config_file) as f:
        planner_params = yaml.load(f, Loader=yaml.FullLoader)
    params['variance_bias'] = float(planner_params['variance_bias'])
    params['num_paths'] = int(planner_params['num_paths'])
    params['num_worlds'] = int(planner_params['num_worlds'])

    return params



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