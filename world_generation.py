import numpy as np
import random
import copy
import math
import yaml

def bernoulli_variance(availability_prob):
    return availability_prob*(1-availability_prob)

def sample_bernoulli_avialability_model(availability_model):
    avails = []
    for avail in availability_model:
        new_avail = max(min(random.gauss(avail, math.sqrt(bernoulli_variance(avail))), 1.0), 0.0)
        avails.append(new_avail)
    return avails

def sample_occupancy(prob):
    if random.random() < prob:
        return 1
    else:
        return 0

### Temporal persistence per Toris, Russell, and Sonia Chernova. "Temporal Persistence Modeling for Object Search." IEEE International Conference on Robotics and Automation (ICRA). 2017.
def persistence_prob(mu, delta_t, last_observation):
    if last_observation == 1:
        return math.exp(-(1.0/mu)*(delta_t))
    else:
        return 1.0 - math.exp(-(1.0/mu)*(delta_t))



### Generate base availability models with corresponding Bernoulli variance. To be used as planner models or sampled to produce "true" simulator model.
def generate_window_base_availability_models_with_bernoulli_variance(node_requests, start_time, availability_percent, budget, time_interval, availability_length, availability_chance):

    # generate base availability model with corresponding variance
    base_availability_models = {}
    model_variances = {}
    for request in node_requests:   

        available_time = budget*availability_percent
        num_windows = max(1, int(round(float(available_time)/availability_length)))
        ave_window_offset = float(budget - available_time)/num_windows

        max_shift = ave_window_offset*.1
        max_additional_spread = availability_length*.1

        initial_shift = int(start_time + random.random()*availability_length/2.0)
        window_high = min(int(initial_shift + availability_length + random.random()*2*max_additional_spread - max_additional_spread), start_time + budget)
        old_window_high = window_high
        windows = [[initial_shift, window_high]]
        for window in range(num_windows):
            window_low = max(start_time, int(old_window_high + random.random()*2*max_shift - max_shift))
            window_high = min(int(window_low + availability_length + random.random()*2*max_additional_spread - max_additional_spread), start_time + budget)
            old_window_high = window_high
            windows.append([window_low, window_high])
        def window_check(x, windows):
            available = 1.0 - availability_chance
            for window in windows:
                if (window_low <= x <= window_high):
                    available = availability_chance
            return available
        
        avails = []
        variances = []
        t = start_time
        num_intervals = int((budget - start_time)/time_interval)
        for i in range(num_intervals):
            avail = window_check(t, windows)
            avails.append(avail)
            variances.append(bernoulli_variance(avail))
            t += time_interval
        base_availability_models[request] = avails
        model_variances[request] = variances

    return base_availability_models, model_variances


### Sample individual model from base model with variance.
def sample_model_parameters(node_requests, base_availability_models, model_variances, sampling_method='gauss'):
    sampled_availability_models = {}
    for request in node_requests:
        new_avails = []
        for i in range(len(base_availability_models[request])):
            old_avail = base_availability_models[request][i]
            variance = model_variances[request][i]

            if sampling_method == 'gauss':
                new_avail = max(min(random.gauss(old_avail, math.sqrt(variance)), 1.0), 0.0)
            new_avails.append(new_avail)

        sampled_availability_models[request] = new_avails

    return sampled_availability_models


### Sample simulated availabilities given "true" model, incorporating temporal consistency
def sample_schedule_from_model(node_requests, availability_models, mu, num_intervals, temporal_consistency=True):
    schedules = {}
    for request in node_requests:
        availability_model = availability_models[request]
        initial_occ = sample_occupancy(availability_model[0])
        prev_occ = initial_occ
        occupancies = [initial_occ]
        for i in range(num_intervals)[1:]:
            if temporal_consistency:
                likelihood = persistence_prob(mu, 1, prev_occ)
                # if prev_occ == 1:
                #     evidence_prob = availability_model[i]
                # else:
                #     evidence_prob = 1 - availability_model[i]
                evidence_prob = likelihood*availability_model[i] + (1.0-likelihood)*(1.0-availability_model[i])
                new_prob = likelihood*availability_model[i]/evidence_prob         # Bayesian update of last observation times occ prior
                occ = sample_occupancy(new_prob)
                occupancies.append(occ)
                prev_occ = occ
            else:
                occ = sample_occupancy(availability_model[i])
                occupancies.append(occ)

        schedules[request] = occupancies

    return schedules



def load_base_models_from_file(base_model_filepath, num_deliveries, availability_percent, stat_run):
    filename = base_model_filepath  + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml"
    with open(filename) as f:
        in_dict = yaml.load(f, Loader=yaml.FullLoader)
    node_requests = in_dict['node_requests']
    base_availability_models = {}
    base_model_variances = {}
    for node_request in node_requests:
        avails = []
        variances = []
        for i in range(len(in_dict['base_availability_models'][node_request])):
            avails.append(float(in_dict['base_availability_models'][node_request][i]))
            variances.append(float(in_dict['base_model_variances'][node_request][i]))
        base_availability_models[node_request] = avails
        base_model_variances[node_request] = variances              
    return base_availability_models, base_model_variances, node_requests

def load_schedules_from_file(schedule_filepath, num_deliveries, availability_percent, stat_run):
    filename = schedule_filepath + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml"
    with open(filename) as f:
        in_dict = yaml.load(f, Loader=yaml.FullLoader)
    node_requests = in_dict['node_requests']
    true_availability_models = {}
    true_schedules = {}
    for node_request in node_requests:
        avails = []
        schedules = []
        for i in range(len(in_dict['true_availability_models'][node_request])):
            avails.append(float(in_dict['true_availability_models'][node_request][i]))
            schedules.append(int(in_dict['true_schedules'][node_request][i]))
        true_availability_models[node_request] = avails
        true_schedules[node_request] = schedules    

    return true_availability_models, true_schedules
    

def save_base_models_to_file(base_model_filepath, base_availability_models, base_model_variances, node_requests, num_deliveries, availability_percent, stat_run):
    filename = base_model_filepath + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml"
    out_dict = {}
    out_dict['base_availability_models'] = base_availability_models
    out_dict['base_model_variances'] = base_model_variances
    out_dict['node_requests'] = node_requests
    with open(filename, 'w') as f:
         yaml.dump(out_dict, f)

def save_schedules_to_file(schedule_filepath, true_availability_models, true_schedules, node_requests, num_deliveries, availability_percent, stat_run):
    filename = schedule_filepath + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml"
    out_dict = {}
    out_dict['true_availability_models'] = true_availability_models
    out_dict['true_schedules'] = true_schedules
    out_dict['node_requests'] = node_requests
    with open(filename, 'w') as f:
        yaml.dump(out_dict, f)