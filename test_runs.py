import numpy as np
import random
import copy
import math
from utils import load_params, load_scenario
from td_op import Graph
from world_generation import generate_window_base_availability_models_with_bernoulli_variance, sample_model_parameters, sample_schedule_from_model, save_base_models_to_file, save_schedules_to_file, load_base_models_from_file, load_schedules_from_file
from plan_execution_simulation import plan_and_execute
from planners import visualize_path_willow


def stat_runs(world_config_file, schedule_config_file, planner_config_file, base_model_filepath, schedule_filepath, out_csv, out_img, strategies, num_deliveries_runs, availability_percents, num_stat_runs, record_output, generate_schedules):

    ## params
    params = load_params(param_config_file)
    

    ## import world
    rooms, start_node_id, maintenance_node, start_time, max_rooms, graph_filename = load_scenario(scenario_config_file)
    g = Graph()
    g.read_graph_from_file(graph_filename)

    
   
    
    for num_deliveries in num_deliveries_runs:
        for availability_percent in availability_percents:

            # temporal consistency parameter
            if params['availabilities'] == 'windows':
                available_time = params['budget']*availability_percent
                num_windows = int(round(float(available_time)/params['availability_length']))
                ave_window_offset = float(params['budget'] - available_time)/num_windows
                mu = ave_window_offset
            else:
                mu = 60


            # base models, true schedules
            if generate_schedules:

                # sample rooms for delivieries 
                node_requests = []
                for stat_run in range(num_stat_runs):
                    if params['node_closeness'] == 'random':
                        node_requests.append(random.sample(rooms, num_deliveries))
                    if params['node_closeness'] == 'sequential':
                        node_requests.append(rooms[0:num_deliveries])

                
                base_availability_models = []
                base_model_variances = []
                true_availability_models = []
                true_schedules = []
                for stat_run in range(num_stat_runs):
                    ## base availability models
                    avails, variances = generate_window_base_availability_models_with_bernoulli_variance(node_requests[stat_run], start_time, availability_percent, params['budget'], params['time_interval'], params['availability_length'], params['availability_chance'])
                    base_availability_models.append(avails)
                    base_model_variances.append(variances)
                        
                    ## true availability models
                    sampled_avails = sample_model_parameters(node_requests[stat_run], avails, variances, params['sampling_method'])
                    true_availability_models.append(sampled_avails)


                    ## true schedules
                    true_schedules.append(sample_schedule_from_model(node_requests[stat_run], sampled_avails, mu, params['num_intervals'], params['temporal_consistency']))

                    save_base_models_to_file(base_model_filepath, base_availability_models[stat_run], base_model_variances[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)
                    save_schedules_to_file(schedule_filepath, true_availability_models[stat_run], true_schedules[stat_run], node_requests[stat_run], num_deliveries, availability_percent, stat_run)

            else:
                # load pre-generated schedules/models
                node_requests = []
                base_availability_models = []
                base_model_variances = []
                true_availability_models = []
                true_schedules = []
                for stat_run in range(num_stat_runs):
                    base_avails, base_variances, requests = load_base_models_from_file(base_model_filepath, num_deliveries, availability_percent, stat_run)
                    true_avails, schedules = load_schedules_from_file(schedule_filepath, num_deliveries, availability_percent, stat_run)
                    node_requests.append(requests)
                    base_availability_models.append(base_avails)
                    base_model_variances.append(base_variances)
                    true_availability_models.append(true_avails)
                    true_schedules.append(schedules)




            ## "learned" availability models
            availability_models = base_availability_models
            model_variances = base_model_variances




            for strategy in strategies:

                competitive_ratios = []
                maintenance_competitive_ratios = []

                for stat_run in range(num_stat_runs):
                    total_profit, competitive_ratio, maintenance_competitive_ratio, path_history = plan_and_execute(strategy, g, availability_models[stat_run], model_variances[stat_run], true_schedules[stat_run], node_requests[stat_run], start_time, start_node_id, maintenance_node, mu, params)
                    competitive_ratios.append(competitive_ratio)
                    maintenance_competitive_ratios.append(maintenance_competitive_ratio)

                # competitive_ratio_ave = np.mean(np.array(competitive_ratios))
                # competitive_ratio_stdev = np.std(np.array(competitive_ratios))
                # maintenance_competitive_ratio_ave = np.mean(np.array(maintenance_competitive_ratios))
                # maintenance_competitive_ratio_stdev = np.std(np.array(maintenance_competitive_ratios))

                    if record_output:
                        with open(out_csv, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([strategy, budget, num_deliveries, availability_percent, params.availability_chance, params.maintenance_reward, params.max_noise_amplitude, params.variance_bias, competitive_ratio, maintenance_competitive_ratio])




def vizualize_sample_execution(param_config_file, scenario_config_file, base_model_filepath, schedule_filepath, out_csv, out_img, strategies, num_deliveries, availability_percent, num_stat_runs, record_output, generate_schedules):

    ## params
    params = load_params(param_config_file)
    

    ## import world
    rooms, start_node_id, maintenance_node, start_time, max_rooms, graph_filename = load_scenario(scenario_config_file)
    g = Graph()
    g.read_graph_from_file(graph_filename)

    

    # temporal consistency parameter
    if params['availabilities'] == 'windows':
        available_time = params['budget']*availability_percent
        num_windows = int(round(float(available_time)/params['availability_length']))
        ave_window_offset = float(params['budget'] - available_time)/num_windows
        mu = ave_window_offset
    else:
        mu = 60


    # base models, true schedules
    if generate_schedules:

        # sample rooms for delivieries 
        node_requests = []
        if node_closeness == 'random':
            node_requests = random.sample(rooms, num_deliveries)
        if node_closeness == 'sequential':
            node_requests = rooms[0:num_deliveries]

        
        base_availability_models = []
        base_model_variances = []
        true_availability_models = []
        true_schedules = []

        ## base availability models
        base_availability_models, base_model_variances = generate_window_base_availability_models_with_bernoulli_variance(node_requests, start_time, availability_percent, params['budget'], params['time_interval'], params['availability_length'])
            
        ## true availability models
        true_availability_models = sample_model_parameters(node_requests, avails, variances, params['sampling_method'])

        ## true schedules
        true_schedules = sample_schedule_from_model(node_requests, true_availability_models, mu, params['num_intervals'])

        save_base_models_to_file(base_model_filepath, base_availability_models, base_model_variances, node_requests, num_deliveries, availability_percent, 0)
        save_schedules_to_file(schedule_filepath, true_availability_models, true_schedules, node_requests, num_deliveries, availability_percent, 0)

    else:
        # load pre-generated schedules/models
        base_availability_models, base_model_variances, node_requests = load_base_models_from_file(base_model_filepath, num_stat_runs, num_deliveries, availability_percent, 0)
        true_availability_models, true_schedules = load_schedules_from_file(schedule_filepath, node_requests, num_stat_runs, num_deliveries, availability_percent, 0)


    ## "learned" availability models
    availability_models = base_availability_models
    model_variances = base_model_variances


    visit_traces = {}
    for strategy in strategies:
        total_profit, competitive_ratio, maintenance_competitive_ratio, path_history = plan_and_execute(strategy, g, availability_models, model_variances, true_schedules, node_requests, start_time, start_node_id, maintenance_node, mu, params)
        visit_traces[strategy] = path_history

    visualize_path_willow(strategies, visit_traces, true_availability_models, true_schedules, node_requests, maintenance_node, start_time, params['budget'], params['time_interval'], out_img)