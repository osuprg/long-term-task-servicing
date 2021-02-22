import numpy as np
import random
import copy
import math
import networkx as nx  
from planners import plan_path
from mcts import MCTS
from utils import visualize_graph
import imageio
from timeit import default_timer as timer



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
    action_history = []
    curr_time = params['start_time']
    curr_node = params['start_node_id']
    path_length = 1
    path_visits = 0
    plan_times = []

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

        runtime_start = timer()
        path = plan_path(strategy, g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
        path_length = len(path)
        plan_time = timer() - runtime_start
        plan_times.append(plan_time)
        print ("Plan time: " + str(plan_time))

        ## Execute
        if path_length > 1:

            path_visits = 1
            for next_step in path:
                visit = next_step[0]
                action = next_step[1]
                dist = next_step[2]
                path_history.append(visit)
                action_history.append(action + '_' + visit + '_' + str(curr_time))
                # if curr_node == visit:
                #     dist = 1
                # else:
                #     # dist = g.get_distance(curr_node, visit)
                #     dist = g[curr_node][visit]['weight']
                curr_time += dist
                curr_node = visit
                path_visits += 1

                if (visit == params['maintenance_node']) and (action == 'maintenance'):
                    total_maintenance_profit += params['maintenance_reward']

                breakout = False
                curr_time_index = int(curr_time/params['time_interval'])
                if curr_time_index > (params['num_intervals'] - 1):
                    print("Curr time index exceeds num intervals: " + str(curr_time_index))
                    curr_time_index = params['num_intervals']-1
                    path_visits = 0
                    path_length = 0
                    break
                # assert(curr_time_index <= (params['num_intervals'] - 1))

                if visit in requests_left_to_deliver:
                    if action == 'deliver':
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
                            # observation
                            availability_observations[visit] = [0, curr_time]

                            # put package back
                            if (curr_time + dist/2) <= (params['start_time'] + params['budget']):
                                curr_time += dist/2
                            else:
                                curr_time = params['start_time'] + params['budget']
                            curr_node = params['start_node_id']
                            path_visits += 1
                            path_history.append(curr_node)

                            if replan:
                                path_visits = 0
                                path_length = 1
                                breakout = True

                    # Break out after every observation
                    if action == 'observe':
                        available = true_schedules[visit][curr_time_index]
                        availability_observations[visit] = [available, curr_time]
                        if replan:
                            path_visits = 0
                            path_length = 1
                            breakout = True

                if visualize:
                    img = visualize_graph(g, base_availability_models, true_schedules, availability_observations, curr_time_index, curr_node, node_requests, nodes_delivered, curr_time, mu, strategy, params['use_gp'])
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

    print ()
    print(strategy + " cr: " + str(competitive_ratio))

    ave_plan_time = sum(plan_times)/len(plan_times)

    if visualize:
        imageio.mimsave(visualize_path, img_history, duration=2)
        print (action_history)

    return total_profit, competitive_ratio, maintenance_competitive_ratio, path_history, ave_plan_time




def create_policy_and_execute(strategy, g, base_availability_models, base_model_variances, true_schedules, node_requests, mu, params, visualize, visualize_path):

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
    action_history = []
    curr_time = params['start_time']
    curr_node = params['start_node_id']
    # path_length = path_length       # FIXME
    path_visits = 0
    maintenance_reward_collected_current_plan = 0.0
    failure_penalty_current_plan = 0.0
    plan_times = []

    # # replan
    # if (strategy == 'no_temp') or (strategy == 'no_replan'):
    #     replan = False
    # else:
    #     replan = True

    # # multiple visits planned
    # if (strategy == 'observe_sampling_mult_visits') or ('observe_mult_visits'):
    #     multiple_visits = True
    # else:
    #     multiple_visits = False


    # if visualize:
        # visualize_graph(g, base_availability_models, true_schedules, availability_observations, curr_time_index, curr_node, node_requests, delivery_history, curr_time, mu)

        # img_history.append(img)

    replan = True
    end_reached = False


    while not(end_reached):

        if replan:
            if curr_time >= (params['start_time'] + params['budget']):
                end_reached = True
                break


            runtime_start = timer()
            mcts = MCTS(g, base_availability_models, availability_observations, requests_left_to_deliver, curr_node, curr_time, params['budget']-curr_time, params['max_iterations'], params['planning_horizon'], params['maintenance_reward'], params['deliver_reward'], params['mu'], params['discovery_factor'], params['distribution_node'], params['maintenance_node'], params['ensemble_method'])
            mcts.create_policy()
            plan_time = timer() - runtime_start
            plan_times.append(plan_time)
            print ("MCTS plan time: " + str(plan_time))

            path_visits = 1
            curr_state = mcts.root_node_id
            maintenance_reward_collected_current_plan = 0.0
            failure_penalty_current_plan = 0.0
            replan = False

        else:
            # follow policy
            next_step = mcts.choose_best_action(curr_state, params['min_expansions'], maintenance_reward_collected_current_plan, failure_penalty_current_plan)

            # print (next_step)

            # did not explore or not enough visits
            if next_step == None:
                replan = True
                print ('Replanning')
                continue
            else:
                
                action = next_step[0]
                visits = next_step[2]
                distances = next_step[3]

                # print ('Curr node: ' + curr_node)
                # print ('Time: ' + str(curr_time))
                # print ('Action: ' + action)
                # print ()

                if action == 'move':
                    curr_state = visits[0]
                    visit = mcts.nodes[curr_state].pose_id
                    dist = distances[0]
                    action_history.append('move_' + visit + '_' + str(curr_time))
                    curr_time += dist
                    curr_node = visit
                    path_visits += 1
                    path_history.append(visit)

                elif action == 'maintenance':
                    curr_state = visits[0]
                    visit = mcts.nodes[curr_state].pose_id
                    dist = distances[0]
                    action_history.append('maintenance_' + visit + '_' + str(curr_time))
                    curr_time += dist
                    curr_node = visit
                    path_visits += 1
                    path_history.append(visit)
                    total_maintenance_profit += params['maintenance_reward']
                    maintenance_reward_collected_current_plan += params['maintenance_reward']
                    
                elif action == 'observe':
                    visit = mcts.nodes[visits[0]].pose_id
                    curr_time_index = int(curr_time/params['time_interval'])
                    available = true_schedules[visit][curr_time_index]
                    availability_observations[visit] = [available, curr_time]
                    if bool(available):
                        curr_state = visits[0]
                        dist  = distances[0]
                    else:
                        curr_state = visits[1]
                        dist = distances[1]
                        visit = mcts.nodes[visits[1]].pose_id

                    action_history.append('observe_' + visit + '_' + str(curr_time))
                    curr_time += dist
                    curr_node = visit
                    path_visits += 1
                    path_history.append(visit)

                elif action == 'deliver':
                    visit = mcts.nodes[visits[0]].pose_id
                    deliver_time = curr_time + distances[0]
                    if deliver_time > (params['start_time'] + params['budget']):
                        end_reached = True
                        break

                    deliver_time_index = int(deliver_time/params['time_interval'])
                    available = true_schedules[visit][deliver_time_index]
                    availability_observations[visit] = [available, deliver_time]
                    if bool(available):
                        curr_state = visits[0]
                        dist  = distances[0]
                        requests_left_to_deliver.remove(visit)
                        total_profit += params['deliver_reward']
                        delivery_history.append([visit, deliver_time])
                        nodes_delivered.append(visit)
                    else:
                        curr_state = visits[1]
                        dist = distances[1]
                        visit = mcts.nodes[visits[1]].pose_id
                        failure_penalty_current_plan -= params['deliver_reward']


                    action_history.append('deliver_' + visit + '_' + str(curr_time))
                    curr_time += dist
                    curr_node = visit
                    path_visits += 1
                    path_history.append(visit)
                    

                

            if curr_time >= (params['start_time'] + params['budget']):
                end_reached = True
                break



            if visualize:
                curr_time_index = int(curr_time/params['time_interval'])
                img = visualize_graph(g, base_availability_models, true_schedules, availability_observations, curr_time_index, curr_node, node_requests, nodes_delivered, curr_time, params['mu'], strategy, params['use_gp'])
                img_history.append(img)

            # if breakout:
            #     break
 




    ratio_divisor = num_requests*params['deliver_reward'] + ((params['budget']-params['start_time']-num_requests)/params['time_interval'])*params['maintenance_reward']
    competitive_ratio = (float(total_profit) + total_maintenance_profit)/ratio_divisor
    maintenance_ratio_divisor = ((params['budget']-params['start_time'])/params['time_interval'])*params['maintenance_reward']
    maintenance_competitive_ratio = total_maintenance_profit/maintenance_ratio_divisor

    print ()
    print(strategy + " cr: " + str(competitive_ratio))

    ave_plan_time = sum(plan_times)/len(plan_times)

    if visualize:
        imageio.mimsave(visualize_path, img_history, duration=2)
        print (action_history)

    return total_profit, competitive_ratio, maintenance_competitive_ratio, path_history, ave_plan_time
