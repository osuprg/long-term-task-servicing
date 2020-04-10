import numpy as np
import random
import copy
import math



def plan_and_execute(strategy, g, base_availability_models, true_schedules, node_requests, start_time, start_node_id, maintenance_node, mu, params):

    ### plan, execution loop
    num_requests = len(node_requests)
    requests_left_to_deliver = copy.deepcopy(node_requests)
    availability_observations = {}
    total_profit = 0.0
    total_maintenance_profit = 0.0
    delivery_history = []
    path_history = [start_node_id]
    curr_time = start_time
    curr_node = start_node_id
    path_length = 1
    path_visits = 0

    # replan
    if (strategy == 'no_temp') or (strategy == 'no_replan'):
        replan = False
    else:
        replan = True

    # multiple visits planned
    if strategy == 'observe_sampling_mult_visits':
        multiple_visits = True
    else:
        multiple_visits = False




    while (path_visits < path_length):

        # runtime_start = timer()
        path = plan_path(strategy, g, base_availability_models, availability_observations, requests_left_to_deliver, curr_time, curr_node, maintenance_node, mu, params)
        # plan_time = timer() - runtime_start
        # print ("Plan time: " + str(plan_time))

        ### Execute
        path_visits = 1
        for visit in path[1:]:
            path_history.append(visit)
            if curr_node == visit:
                dist = 1
            else:
                dist = g.get_distance(curr_node, visit)
            curr_time += dist
            curr_node = visit
            path_visits += 1

            if visit == maintenance_node:
                total_maintenance_profit += maintenance_reward

            if visit in requests_left_to_deliver:
                curr_time_index = int(curr_time/params['time_interval'])
                available = true_schedules[visit][curr_time_index]
                if available:
                    requests_left_to_deliver.remove(visit)
                    total_profit += deliver_reward
                    delivery_history.append([visit, curr_time])
                    # availability_observations[visit] = [1, curr_time]

                    if multiple_visits:
                        if replan:
                            path_visits = 0
                            path_length = 1
                            break
                else:
                    if replan:
                        availability_observations[visit] = [0, curr_time]
                        path_visits = 0
                        path_length = 1
                        break


    ratio_divisor = num_requests*params.deliver_reward + ((params['budget']-start_time-num_requests)/params['time_interval'])*params.maintenance_reward
    competitive_ratio = (float(total_profit) + total_maintenance_profit)/ratio_divisor
    maintenance_ratio_divisor = ((params['budget']-start_time)/params['time_interval'])*maintenance_reward
    maintenance_competitive_ratio = total_maintenance_profit/maintenance_ratio_divisor

    return total_profit, competitive_ratio, maintenance_competitive_ratio, path_history
    