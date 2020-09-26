# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
import os.path
import csv
import networkx as nx  
from utils import load_params, read_graph_from_file, load_brayford_training_data, load_brayford_testing_data, load_brayford_training_data_histogram, load_brayford_testing_data_histogram
# from gp import GP
from td_op import Graph
from world_generation import generate_graph
from schedule_generation import generate_windows_overlapping, generate_windows, generate_window_base_availability_models_with_bernoulli_variance, sample_model_parameters, generate_schedule, save_base_models_to_file, save_schedules_to_file, load_base_models_from_file, load_schedules_from_file, generate_simple_models, generate_simple_schedules
from plan_execution_simulation import plan_and_execute, create_policy_and_execute
from planners import visualize_path_willow
from spectral_clustering import build_gmm


### High level code for running stat runs of task planning and simulated execution
def stat_runs(world_config_file, schedule_config_file, planner_config_file, model_config_file, base_model_filepath, schedule_filepath, output_file, strategies, num_deliveries_runs, availability_percents, budgets, num_stat_runs, visualize, out_gif_path):

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
    g, rooms = generate_graph(params['graph_generator_type'], os.path.dirname(os.path.abspath(__file__)), params['graph_filename'], params['max_rooms'], params['rooms'], params['max_traversal_cost'], params['distance_scaling'])
    params['rooms'] = rooms
    
    # for num_deliveries in num_deliveries_runs:
    num_deliveries = num_deliveries_runs[0]
    for availability_percent in availability_percents:

        for budget in budgets:

            params['budget'] = budget
            params['longest_period'] = budget

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
            params['mu'] = mu


            # base models, true schedules
            node_requests = []
            base_availability_models = []
            base_model_variances = []
            true_availability_models = []
            true_schedules = []
            num_test_runs = 0
            for stat_run in range(num_stat_runs):
                model_file_exists = os.path.exists(base_model_filepath  + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
                schedule_file_exists = os.path.exists(schedule_filepath + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
                if model_file_exists and schedule_file_exists:
                #     # load pre-generated schedules/models
                    base_avails, base_variances, requests = load_base_models_from_file(base_model_filepath, num_deliveries, availability_percent, stat_run)
                    true_avails, schedules = load_schedules_from_file(schedule_filepath, num_deliveries, availability_percent, stat_run)
                    node_requests.append(requests)
                    base_availability_models.append(base_avails)
                    base_model_variances.append(base_variances)
                    true_availability_models.append(true_avails)
                    true_schedules.append(schedules)
                else:

                    if params['availabilities'] == 'brayford':

                        # model
                        if params['use_gp']:
                            from gp import GP
                            gps = {}
                        # if params['use_gmm']:
                        gmms = {}
                        # mus = {}
                        mu = 0.0
                        mu_n = 0
                        node_requests.append(params['rooms'])
                        for request in node_requests[stat_run]:
                            x_in, y_in, mu_combined, mu_combined_n = load_brayford_training_data(request, os.path.dirname(os.path.abspath(__file__)) + params['data_path'], out_gif_path)
                            if params['use_gp']:
                                gps[request] = GP(None, x_in, y_in, params['budget'], 1, params['noise_scaling'], True, 'values')
                            else:
                                gmms[request] = build_gmm(x_in, y_in, params['start_time'], params['start_time'] + params['budget'], params['time_interval'], params)
                                # gmms[request].visualize(out_gif_path + "train_" + request + "_gmm_histogram_10.jpg", request)
                            # mus[request] = mu_combined/mu_combined_n
                            mu += mu_combined
                            mu_n += mu_combined_n

                            # gps[request].visualize(out_gif_path + "train_" + request + "_model_histogram_10.jpg", request)

                        if params['use_gp']:
                            base_availability_models.append(gps)
                        else:
                            base_availability_models.append(gmms)
                        base_model_variances.append({})
                        mu = mu/mu_n
                        params['mu'] = mu


                        # true schedule
                        # if params['availabilities'] == 'brayford':

                        schedules = {}
                        for request in node_requests[stat_run]:
                            X, Y = load_brayford_testing_data(request, os.path.dirname(os.path.abspath(__file__)) + params['data_path'], stat_run, out_gif_path)
                            for i in range(Y.shape[0]):
                                if not(i in schedules):
                                    schedules[i] = {}
                                schedules[i][request] = Y[i]
                            num_test_runs = Y.shape[0]
                            # schedules[request] = Y
                            # if params['use_gp']:
                            # from gp import GP
                            # test_gp = GP(None, x_in, y_in, params['budget'], 1, params['noise_scaling'], True, 'values')
                            # if params['use_gmm']:
                            #     test_gp = build_gmm(x_in, y_in, params['start_time'], params['start_time'] + params['budget'], params['time_interval'], params)
                            # if stat_run == 0:
                            #     test_gp.visualize(out_gif_path + "february_" + request + "_model_10.jpg", request)
                            # else:
                            # test_gp.visualize(out_gif_path + "november_" + request + "_model_histogram_10.jpg", request)
                            # schedules[request] = test_gp.threshold_sample_schedule(params['start_time'], params['budget'], params['time_interval'])

                            # # visualize:
                            # fig = plt.figure()
                            # X = np.array(list(range(params['start_time'], params['budget'], params['time_interval'])))
                            # Y = np.array(schedules[request])
                            # plt.scatter(X, Y)
                            # if stat_run == 0:
                            #     plt.title("Brayford Schedule Node " + request + ": February")
                            #     plt.savefig(out_gif_path + "february_" + request + ".jpg")
                            # else:
                            #     plt.title("Brayford Schedule Node " + request + ": November")
                            #     plt.savefig(out_gif_path + "november_" + request + ".jpg")

                        true_schedules.append(schedules)


                    elif params['availabilities'] == 'windows': 

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
                                y_in = copy.deepcopy(avails[request])
                                for i in range(len(y_in)):
                                    y = max(y_in[i]+random.random()*params['noise_scaling'] - params['noise_scaling']/2.0, 0.01)
                                    y = min(y, .99)
                                    y_in[i] = y

                                gps[request] = GP(None, x_in, y_in, params['budget'], params['spacing'], 0.0, True, 'values')
                            base_availability_models.append(gps)
                        else:
                            gmms = {}
                            for request in node_requests[stat_run]:
                                x_in = list(range(params['start_time'], params['budget'], params['time_interval']))
                                y_in = copy.deepcopy(avails[request])
                                for i in range(len(y_in)):
                                    y = max(y_in[i]+random.random()*params['noise_scaling'] - params['noise_scaling']/2.0, 0.01)
                                    y = min(y, .99)
                                    y_in[i] = y
                                gmms[request] = build_gmm(x_in, y_in, params['start_time'], params['start_time'] + params['budget'], params['time_interval'], params, True)
                                # gmms[request].visualize(out_gif_path + "train_" + request + "_gmm_histogram_10.jpg", request)
                                # mus[request] = mu_combined/mu_combined_n
                                # mu += mu_combined
                                # mu_n += mu_combined_n
                            base_availability_models.append(gmms)
                        # else:
                        #     base_availability_models.append(avails)

                        # base_availability_models.append(avails)
                        base_model_variances.append(variances)

                        # true availability models
                        sampled_avails = sample_model_parameters(node_requests[stat_run], avails, variances, params['sampling_method'])
                        sampled_avails = avails
                        

                        true_availability_models.append(avails)

                        ## true schedules
                        true_schedules.append(generate_schedule(node_requests[stat_run], avails, params['mu'], params['num_intervals'], params['schedule_generation_method'], params['temporal_consistency']))
                        # true_schedules.append(sample_schedule_from_model(node_requests[stat_run], sampled_avails, mu, params['num_intervals'], params['temporal_consistency']))

                        # save_base_models_to_file(base_model_filepath, base_availability_models[stat_run], base_model_variances[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                        # save_schedules_to_file(schedule_filepath, true_availability_models[stat_run], true_schedules[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)


                    # elif params['availabilities'] == 'simple':
                            
                    #     # sample rooms for delivieries 
                    #     if params['node_closeness'] == 'random':
                    #         node_requests.append(random.sample(params['rooms'], num_deliveries))
                    #     if params['node_closeness'] == 'sequential':
                    #         node_requests.append(params['rooms'][0:num_deliveries])
                        
                    #     ## base availability models
                    #     avails, variances = generate_simple_models(node_requests[stat_run], params['start_time'], availability_percent, params['budget'], params['time_interval'], params['availability_length'], params['availability_chance'])
                    #     base_availability_models.append(avails)
                    #     base_model_variances.append(variances)
                            
                    #     # ## true availability models
                    #     # sampled_avails = sample_model_parameters(node_requests[stat_run], avails, variances, params['sampling_method'])
                    #     # true_availability_models.append(sampled_avails)

                    #     ## true schedules
                    #     true_schedules.append(generate_simple_schedules(node_requests[stat_run], sampled_avails, params['mu'], params['num_intervals'], params['schedule_generation_method']))
                    #     # true_schedules.append(sample_schedule_from_model(node_requests[stat_run], sampled_avails, mu, params['num_intervals'], params['temporal_consistency']))

                    #     # save_base_models_to_file(base_model_filepath, base_availability_models[stat_run], base_model_variances[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                    #     # save_schedules_to_file(schedule_filepath, true_availability_models[stat_run], true_schedules[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                        
                    else:
                        raise ValueError(params['availabilities'])
                    

            ## "learned" availability models
            availability_models = base_availability_models
            model_variances = base_model_variances


            # plan and execute paths for specified strategies
            for strategy in strategies:

                strategy_name = strategy

                params['uncertainty_penalty'] = 0.0
                params['observation_reward'] = 0.0
                params['deliver_threshold'] = 0.0
                

                if strategy == 'observe_mult_visits_up_5_or_0_dt_0':
                    params['uncertainty_penalty'] = 0.5
                    params['observation_reward'] = 0.0
                    params['deliver_threshold'] = 0.0
                    strategy_name = strategy
                    strategy = 'observe_mult_visits'

                if strategy == 'observe_mult_visits_up_0_or_7_dt_0':
                    params['uncertainty_penalty'] = 0.0
                    params['observation_reward'] = 0.7
                    params['deliver_threshold'] = 0.0
                    strategy_name = strategy
                    strategy = 'observe_mult_visits'

                if strategy == 'observe_mult_visits_up_5_or_7_dt_0':
                    params['uncertainty_penalty'] = 0.5
                    params['observation_reward'] = 0.7
                    params['deliver_threshold'] = 0.0
                    strategy_name = strategy
                    strategy = 'observe_mult_visits'


                


                for stat_run in range(num_stat_runs):
                # stat_run = 0
                # for test_run in range(num_test_runs):
                    if strategy == 'mcts':
                        total_profit, competitive_ratio, maintenance_competitive_ratio, path_history, ave_plan_time = create_policy_and_execute(strategy, g, availability_models[stat_run], model_variances[stat_run], true_schedules[stat_run], node_requests[stat_run], params['mu'], params, visualize, out_gif_path)
                    else:
                        total_profit, competitive_ratio, maintenance_competitive_ratio, path_history, ave_plan_time = plan_and_execute(strategy, g, availability_models[stat_run], model_variances[stat_run], true_schedules[stat_run], node_requests[stat_run], params['mu'], params, visualize, out_gif_path)
                    
                    if record_output:
                        with open(output_file, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([strategy_name, params['budget'], num_deliveries, availability_percent, params['availability_chance'], params['maintenance_reward'], params['max_noise_amplitude'], params['variance_bias'], competitive_ratio, maintenance_competitive_ratio, ave_plan_time])




### High level function for visualizing simulated execution
def vizualize_sample_execution(world_config_file, schedule_config_file, planner_config_file, model_config_file, base_model_filepath, schedule_filepath, strategies, num_deliveries_runs, availability_percents, stat_run, visualize, out_gif_path, out_img_path):

    ## params
    params = load_params(world_config_file, schedule_config_file, planner_config_file, model_config_file)
    

    ## import world
    # g = Graph()
    # g.read_graph_from_file(os.path.dirname(os.path.abspath(__file__)) + params['graph_filename'])

    # g = read_graph_from_file(os.path.dirname(os.path.abspath(__file__)) + params['graph_filename'])
    g, rooms = generate_graph(params['graph_generator_type'], os.path.dirname(os.path.abspath(__file__)), params['graph_filename'], params['max_rooms'], params['rooms'], params['max_traversal_cost'], params['distance_scaling'])
    params['rooms'] = rooms

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
            params['mu'] = mu


            # base models, true schedules
            stat_run = 0
            model_file_exists = os.path.exists(base_model_filepath  + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
            schedule_file_exists = os.path.exists(schedule_filepath + str(num_deliveries) + "_" + str(availability_percent) + "_" + str(stat_run) + ".yaml")
            if model_file_exists and schedule_file_exists:
                # load pre-generated schedules/models
                base_availability_models, base_model_variances, node_requests = load_base_models_from_file(base_model_filepath, num_deliveries, availability_percent, stat_run)
                true_availability_models, true_schedules = load_schedules_from_file(schedule_filepath, num_deliveries, availability_percent, stat_run)
                availabilities = base_availability_models
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
                        availabilities = {}
                        for request in node_requests:
                            x_in = list(range(params['start_time'], params['budget'], params['time_interval']))
                            gps[request] = GP(None, x_in, avails[request], params['budget'], params['spacing'], params['noise_scaling'], True, 'values')
                            availabilities[request] = gps[request].get_preds(x_in)
                        base_availability_models = gps
                    else:
                        base_availability_models = avails
                        availabilities = avails

                    ## true availability models
                    # sampled_availability_models = sample_model_parameters(node_requests, base_availability_models, base_model_variances, params['sampling_method'])
                    true_availability_models = avails

                    ## true schedules
                    true_schedules = generate_schedule(node_requests, true_availability_models, params['mu'], params['num_intervals'], params['schedule_generation_method'], params['temporal_consistency'])

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
                    availabilities = base_availability_models
                                          
                    # ## true availability models
                    # sampled_avails = sample_model_parameters(node_requests[stat_run], avails, variances, params['sampling_method'])
                    # true_availability_models.append(sampled_avails)

                    ## true schedules
                    true_schedules = generate_simple_schedules(node_requests, base_availability_models, params['mu'], params['num_intervals'], params['schedule_generation_method'])
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
                if strategy == 'mcts':
                    total_profit, competitive_ratio, maintenance_competitive_ratio, path_history = create_policy_and_execute(strategy, g, availability_models, model_variances, true_schedules, node_requests, params['mu'], params, visualize, out_gif_path)
                else:
                    total_profit, competitive_ratio, maintenance_competitive_ratio, path_history = plan_and_execute(strategy, g, availability_models, model_variances, true_schedules, node_requests, params['mu'], params, visualize, out_gif_path)
                visit_traces[strategy] = path_history

            visualize_path_willow(strategies, visit_traces, availabilities, true_schedules, node_requests, params['maintenance_node'], params['start_time'], params['budget'], params['time_interval'], out_img_path)