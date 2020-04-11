import yaml    

def load_params(param_config_file):
    with open(param_config_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['budget'] = int(params['budget'])
    params['time_interval'] = int(params['time_interval'])
    params['num_intervals'] = int(params['budget']/params['time_interval'])
    params['availability_length'] = int(params['availability_length'])
    params['availability_chance'] = float(params['availability_chance'])
    params['temporal_consistency'] = bool(int(params['temporal_consistency']))
    params['maintenance_reward'] = float(params['maintenance_reward'])
    params['deliver_reward'] = float(params['deliver_reward'])
    params['max_noise_amplitude'] = float(params['max_noise_amplitude'])
    params['variance_bias'] = float(params['variance_bias'])
    params['num_paths'] = int(params['num_paths'])
    params['num_worlds'] = int(params['num_worlds'])
    num_deliveries = []
    for n in params['num_deliveries']:
        num_deliveries.append(int(n))
    params['num_deliveries'] = num_deliveries
    availability_percents = []
    for n in params['availability_percents']:
        availability_percents.append(float(n))
    params['availability_percents'] = availability_percents
    return params

def load_scenario(scenario_config_file):
    with open(scenario_config_file) as f:
        scenario = yaml.load(f, Loader=yaml.FullLoader)
    rooms = scenario['rooms']
    start_node_id = scenario['start_node_id']
    maintenance_node = scenario['maintenance_node']
    start_time = int(scenario['start_time'])
    max_rooms = int(scenario['max_rooms'])
    graph_filename = scenario['graph_filename']
    return rooms, start_node_id, maintenance_node, start_time, max_rooms, graph_filename


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