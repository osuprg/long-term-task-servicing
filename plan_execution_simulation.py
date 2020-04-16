import numpy as np
import random
import copy
import math
from planners import plan_path



### Simulate specified planning strategy being employed to given task execution scenario
def plan_and_execute(strategy, g, base_availability_models, base_model_variances, true_schedules, node_requests, mu, params):

    ## plan, execution loop
    num_requests = len(node_requests)
    requests_left_to_deliver = copy.deepcopy(node_requests)
    availability_observations = {}
    total_profit = 0.0
    total_maintenance_profit = 0.0
    delivery_history = []
    path_history = [params['start_node_id']]
    curr_time = params['start_time']
    curr_node = params['start_node_id']
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
        path = plan_path(strategy, g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
        path_length = len(path)
        # plan_time = timer() - runtime_start
        # print ("Plan time: " + str(plan_time))

        ## Execute
        if path_length > 1:

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

                if visit == params['maintenance_node']:
                    total_maintenance_profit += params['maintenance_reward']

                if visit in requests_left_to_deliver:
                    curr_time_index = int(curr_time/params['time_interval'])

                    if curr_time_index > (params['num_intervals'] - 1):
                        print("Curr time index exceeds num intervals: " + str(curr_time_index))
                        curr_time_index = params['num_intervals']-1

                    available = true_schedules[visit][curr_time_index]              #FIXME: list index out of range
                    if available:
                        requests_left_to_deliver.remove(visit)
                        total_profit += params['deliver_reward']
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
        else:
            path_visits = 0
            path_length = 0


    ratio_divisor = num_requests*params['deliver_reward'] + ((params['budget']-params['start_time']-num_requests)/params['time_interval'])*params['maintenance_reward']
    competitive_ratio = (float(total_profit) + total_maintenance_profit)/ratio_divisor
    maintenance_ratio_divisor = ((params['budget']-params['start_time'])/params['time_interval'])*params['maintenance_reward']
    maintenance_competitive_ratio = total_maintenance_profit/maintenance_ratio_divisor

    print(strategy + " cr: " + str(competitive_ratio))

    return total_profit, competitive_ratio, maintenance_competitive_ratio, path_history
    