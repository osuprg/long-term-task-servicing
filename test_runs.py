import numpy as np
import random
import copy
import math
import os.path
import csv
import networkx as nx  
from utils import load_params, read_graph_from_file
# from gp import GP
from td_op import Graph
from world_generation import generate_graph
from schedule_generation import generate_windows_overlapping, generate_windows, generate_window_base_availability_models_with_bernoulli_variance, sample_model_parameters, generate_schedule, save_base_models_to_file, save_schedules_to_file, load_base_models_from_file, load_schedules_from_file, generate_simple_models, generate_simple_schedules
from plan_execution_simulation import plan_and_execute
from planners import visualize_path_willow


### High level code for running stat runs of task planning and simulated execution
def stat_runs(world_config_file, schedule_config_file, planner_config_file, model_config_file, base_model_filepath, schedule_filepath, output_file, strategies, num_deliveries_runs, availability_percents, num_stat_runs, visualize, visualize_path):

    if output_file == None:
        record_output = False
    else:
        record_output = True

    ## params
    params = load_params(world_config_file, schedule_config_file, planner_config_file, model_config_file)

    ## load world
    # g = Graph()
    # g.read_graph_from_file(os.path.dirname(os.path.abspath(__file__)) + params['graph_filename'])

    # g = read_graph_from_file(os.path.dirname(os.path.abspath(__file__)) + params['graph_filename'])
    g = generate_graph(params['graph_generator_type'], os.path.dirname(os.path.abspath(__file__)), params['graph_filename'], params['max_rooms'], params['max_traversal_cost'], params['distance_scaling'])

    
    for num_deliveries in num_deliveries_runs:
        for availability_percent in availability_percents:

            # temporal consistency parameter
            if params['availabilities'] == 'windows':
                # available_time = params['budget']*availability_percent
                # num_windows = max(int(round(float(available_time)/params['availability_length'])), 1)
                # ave_window_offset = float(params['budget'] - available_time)/num_windows
                # mu = max(ave_window_offset, 1)

                available_time = params['budget']*availability_percent
                num_windows = max(1, int(round(float(available_time)/params['availability_length'])))
                # new_availability_length = int(float(available_time)/num_windows)
                ave_window_offset = min(float(params['budget'] - available_time)/num_windows, float(params['budget'] - available_time)/2)
                mu = int(ave_window_offset/2)
                
                # mu = int(params['availability_length']/2)
            elif params['availabilities'] == 'simple':
                mu = int(params['availability_length']/2)
            else:
                mu = 30


            # base models, true schedules
            node_requests = []
            base_availability_models = []
            base_model_variances = []
            true_availability_models = []
            true_schedules = []
            for stat_run in range(num_stat_runs):
                model_file_exists = os.path.exists(base_model_filepath  + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
                schedule_file_exists = os.path.exists(schedule_filepath + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
                if model_file_exists and schedule_file_exists:
                    # load pre-generated schedules/models
                    base_avails, base_variances, requests = load_base_models_from_file(base_model_filepath, num_deliveries, availability_percent, stat_run)
                    true_avails, schedules = load_schedules_from_file(schedule_filepath, num_deliveries, availability_percent, stat_run)
                    node_requests.append(requests)
                    base_availability_models.append(base_avails)
                    base_model_variances.append(base_variances)
                    true_availability_models.append(true_avails)
                    true_schedules.append(schedules)
                else:

                    if params['availabilities'] == 'windows': 

                        # sample rooms for delivieries 
                        if params['node_closeness'] == 'random':
                            node_requests.append(random.sample(params['rooms'], num_deliveries))
                        if params['node_closeness'] == 'sequential':
                            node_requests.append(params['rooms'][0:num_deliveries])
                        
                        ## base availability models
                        avails, variances = generate_windows_overlapping(node_requests[stat_run], params['start_time'], availability_percent, params['budget'], params['time_interval'], params['availability_length'], params['availability_chance'])
                        if params['use_gp']:
                            from gp import GP
                            gps = {}
                            for request in node_requests[stat_run]:
                                x_in = list(range(params['start_time'], params['budget'], params['time_interval']))
                                gps[request] = GP(None, x_in, avails[request], params['budget'], params['spacing'], params['noise_scaling'], True, 'values')
                            base_availability_models.append(gps)
                        else:
                            base_availability_models.append(avails)

                        # base_availability_models.append(avails)
                        base_model_variances.append(variances)

                        ## true availability models
                        # sampled_avails = sample_model_parameters(node_requests[stat_run], avails, variances, params['sampling_method'])
                        # sampled_avails = avails
                        

                        true_availability_models.append(avails)

                        ## true schedules
                        true_schedules.append(generate_schedule(node_requests[stat_run], avails, mu, params['num_intervals'], params['schedule_generation_method'], params['temporal_consistency']))
                        # true_schedules.append(sample_schedule_from_model(node_requests[stat_run], sampled_avails, mu, params['num_intervals'], params['temporal_consistency']))

                        # save_base_models_to_file(base_model_filepath, base_availability_models[stat_run], base_model_variances[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                        # save_schedules_to_file(schedule_filepath, true_availability_models[stat_run], true_schedules[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)


                    elif params['availabilities'] == 'simple':
                            
                        # sample rooms for delivieries 
                        if params['node_closeness'] == 'random':
                            node_requests.append(random.sample(params['rooms'], num_deliveries))
                        if params['node_closeness'] == 'sequential':
                            node_requests.append(params['rooms'][0:num_deliveries])
                        
                        ## base availability models
                        avails, variances = generate_simple_models(node_requests[stat_run], params['start_time'], availability_percent, params['budget'], params['time_interval'], params['availability_length'], params['availability_chance'])
                        base_availability_models.append(avails)
                        base_model_variances.append(variances)
                            
                        # ## true availability models
                        # sampled_avails = sample_model_parameters(node_requests[stat_run], avails, variances, params['sampling_method'])
                        # true_availability_models.append(sampled_avails)

                        ## true schedules
                        true_schedules.append(generate_simple_schedules(node_requests[stat_run], sampled_avails, mu, params['num_intervals'], params['schedule_generation_method']))
                        # true_schedules.append(sample_schedule_from_model(node_requests[stat_run], sampled_avails, mu, params['num_intervals'], params['temporal_consistency']))

                        # save_base_models_to_file(base_model_filepath, base_availability_models[stat_run], base_model_variances[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                        # save_schedules_to_file(schedule_filepath, true_availability_models[stat_run], true_schedules[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                        
                    else:
                        raise ValueError(params['availabilities'])
                    

            ## "learned" availability models
            availability_models = base_availability_models
            model_variances = base_model_variances


            # plan and execute paths for specified strategies
            for strategy in strategies:
                for stat_run in range(num_stat_runs):
                    total_profit, competitive_ratio, maintenance_competitive_ratio, path_history = plan_and_execute(strategy, g, availability_models[stat_run], model_variances[stat_run], true_schedules[stat_run], node_requests[stat_run], mu, params, visualize, visualize_path)
                    
                    if record_output:
                        with open(output_file, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([strategy, params['budget'], num_deliveries, availability_percent, params['availability_chance'], params['maintenance_reward'], params['max_noise_amplitude'], params['variance_bias'], competitive_ratio, maintenance_competitive_ratio])




### High level function for visualizing simulated execution
def vizualize_sample_execution(world_config_file, schedule_config_file, planner_config_file, model_config_file, base_model_filepath, schedule_filepath, strategies, num_deliveries_runs, availability_percents, stat_run, visualize, out_gif_path, out_img_path):

    ## params
    params = load_params(world_config_file, schedule_config_file, planner_config_file, model_config_file)
    

    ## import world
    # g = Graph()
    # g.read_graph_from_file(os.path.dirname(os.path.abspath(__file__)) + params['graph_filename'])

    # g = read_graph_from_file(os.path.dirname(os.path.abspath(__file__)) + params['graph_filename'])
    g = generate_graph(params['graph_generator_type'], os.path.dirname(os.path.abspath(__file__)), params['graph_filename'], params['max_rooms'], params['max_traversal_cost'], params['distance_scaling'])


    for num_deliveries in num_deliveries_runs:
        for availability_percent in availability_percents:

            # temporal consistency parameter
            if params['availabilities'] == 'windows':
                # available_time = params['budget']*availability_percent
                # num_windows = max(int(round(float(available_time)/params['availability_length'])), 1)
                # ave_window_offset = float(params['budget'] - available_time)/num_windows
                # mu = max(ave_window_offset, 1)

                available_time = params['budget']*availability_percent
                num_windows = max(1, int(round(float(available_time)/params['availability_length'])))
                # new_availability_length = int(float(available_time)/num_windows)
                ave_window_offset = min(float(params['budget'] - available_time)/num_windows, float(params['budget'] - available_time)/2)
                mu = int(ave_window_offset/2)

                # mu = int(params['availability_length']/2)
            elif params['availabilities'] == 'simple':
                mu = int(params['availability_length']/2)
            else:
                mu = 30


            # base models, true schedules
            model_file_exists = os.path.exists(base_model_filepath  + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
            schedule_file_exists = os.path.exists(schedule_filepath + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
            if model_file_exists and schedule_file_exists:
                # load pre-generated schedules/models
                base_availability_models, base_model_variances, node_requests = load_base_models_from_file(base_model_filepath, num_deliveries, availability_percent, stat_run)
                true_availability_models, true_schedules = load_schedules_from_file(schedule_filepath, num_deliveries, availability_percent, stat_run)
            else:

                if params['availabilities'] == 'windows':
                    # sample rooms for delivieries 
                    if params['node_closeness'] == 'random':
                        node_requests = random.sample(params['rooms'], num_deliveries)
                    if params['node_closeness'] == 'sequential':
                        node_requests = params['rooms'][0:num_deliveries]
                    
                    ## base availability models
                    avails, base_model_variances = generate_windows_overlapping(node_requests, params['start_time'], availability_percent, params['budget'], params['time_interval'], params['availability_length'], params['availability_chance'])
                    
                    if params['use_gp']:
                        from gp import GP
                        gps = {}
                        for request in node_requests[stat_run]:
                            x_in = list(range(params['start_time'], params['budget'], params['time_interval']))
                            gps[request] = GP(None, x_in, avails[request], params['budget'], params['spacing'], params['noise_scaling'], True, 'values')
                        base_availability_models = gps
                    else:
                        base_availability_models = avails

                    ## true availability models
                    # sampled_availability_models = sample_model_parameters(node_requests, base_availability_models, base_model_variances, params['sampling_method'])
                    true_availability_models = avails

                    ## true schedules
                    true_schedules = generate_schedule(node_requests, true_availability_models, mu, params['num_intervals'], params['schedule_generation_method'], params['temporal_consistency'])

                    # save_base_models_to_file(base_model_filepath, base_availability_models, base_model_variances, node_requests, num_deliveries, availability_percent, stat_run)
                    # save_schedules_to_file(schedule_filepath, true_availability_models, true_schedules, node_requests, num_deliveries, availability_percent, stat_run)


                elif params['availabilities'] == 'simple':
                            
                    # sample rooms for delivieries 
                    if params['node_closeness'] == 'random':
                        node_requests = random.sample(params['rooms'], num_deliveries)
                    if params['node_closeness'] == 'sequential':
                        node_requests = params['rooms'][0:num_deliveries]
                    
                    ## base availability models
                    base_availability_models, base_model_variances = generate_simple_models(node_requests, params['start_time'], availability_percent, params['budget'], params['time_interval'], params['availability_length'], params['availability_chance'])
                                            
                    # ## true availability models
                    # sampled_avails = sample_model_parameters(node_requests[stat_run], avails, variances, params['sampling_method'])
                    # true_availability_models.append(sampled_avails)

                    ## true schedules
                    true_schedules = generate_simple_schedules(node_requests, base_availability_models, mu, params['num_intervals'], params['schedule_generation_method'])
                    # true_schedules.append(sample_schedule_from_model(node_requests[stat_run], sampled_avails, mu, params['num_intervals'], params['temporal_consistency']))

                    # save_base_models_to_file(base_model_filepath, base_availability_models[stat_run], base_model_variances[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                    # save_schedules_to_file(schedule_filepath, true_availability_models[stat_run], true_schedules[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                        
                else:
                    raise ValueError(params['availabilities'])
                    

            ## "learned" availability models
            availability_models = base_availability_models
            model_variances = base_model_variances


            # plan and execute paths for specified strategies
            visit_traces = {}
            for strategy in strategies:
                total_profit, competitive_ratio, maintenance_competitive_ratio, path_history = plan_and_execute(strategy, g, availability_models, model_variances, true_schedules, node_requests, mu, params, visualize, out_gif_path)
                visit_traces[strategy] = path_history

            visualize_path_willow(strategies, visit_traces, base_availability_models, true_schedules, node_requests, params['maintenance_node'], params['start_time'], params['budget'], params['time_interval'], out_img_path)