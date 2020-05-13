import numpy as np
import random
import copy
import math
import networkx as nx  
from planners import plan_path
from utils import visualize_graph
import imageio



### Simulate specified planning strategy being employed to given task execution scenario
def plan_and_execute(strategy, g, base_availability_models, base_model_variances, true_schedules, node_requests, mu, params, visualize, visualize_path):

    ## plan, execution loop
    num_requests = len(node_requests)
    requests_left_to_deliver = copy.deepcopy(node_requests)
    availability_observations = {}
    total_profit = 0.0
    total_maintenance_profit = 0.0
    delivery_history = []
    nodes_delivered = []
    img_history = []
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
    if (strategy == 'observe_sampling_mult_visits') or ('observe_mult_visits'):
        multiple_visits = True
    else:
        multiple_visits = False


    # if visualize:
        # visualize_graph(g, base_availability_models, true_schedules, availability_observations, curr_time_index, curr_node, node_requests, delivery_history, curr_time, mu)

        # img_history.append(img)


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
                    # dist = g.get_distance(curr_node, visit)
                    dist = g[curr_node][visit]['weight']
                curr_time += dist
                curr_node = visit
                path_visits += 1

                if visit == params['maintenance_node']:
                    total_maintenance_profit += params['maintenance_reward']

                breakout = False
                curr_time_index = int(curr_time/params['time_interval'])
                if curr_time_index > (params['num_intervals'] - 1):
                    print("Curr time index exceeds num intervals: " + str(curr_time_index))
                    curr_time_index = params['num_intervals']-1
                assert(curr_time_index <= (params['num_intervals'] - 1))

                if visit in requests_left_to_deliver:
                    available = true_schedules[visit][curr_time_index]
                    if bool(available):
                        requests_left_to_deliver.remove(visit)
                        total_profit += params['deliver_reward']
                        delivery_history.append([visit, curr_time])
                        nodes_delivered.append(visit)
                        availability_observations[visit] = [1, curr_time]

                        if multiple_visits:
                            if replan:
                                path_visits = 0
                                path_length = 1
                                breakout = True
                    else:
                        if replan:
                            availability_observations[visit] = [0, curr_time]
                            path_visits = 0
                            path_length = 1
                            breakout = True

                if visualize:
                    img = visualize_graph(g, base_availability_models, true_schedules, availability_observations, curr_time_index, curr_node, node_requests, nodes_delivered, curr_time, mu, strategy)
                    img_history.append(img)

                if breakout:
                    break
        else:
            path_visits = 0
            path_length = 0


    ratio_divisor = num_requests*params['deliver_reward'] + ((params['budget']-params['start_time']-num_requests)/params['time_interval'])*params['maintenance_reward']
    competitive_ratio = (float(total_profit) + total_maintenance_profit)/ratio_divisor
    maintenance_ratio_divisor = ((params['budget']-params['start_time'])/params['time_interval'])*params['maintenance_reward']
    maintenance_competitive_ratio = total_maintenance_profit/maintenance_ratio_divisor

    print(strategy + " cr: " + str(competitive_ratio))

    if visualize:
        imageio.mimsave(visualize_path, img_history, duration=2)

    return total_profit, competitive_ratio, maintenance_competitive_ratio, path_history
