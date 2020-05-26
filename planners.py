import numpy as np
import random
import copy
import math
import networkx as nx  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from td_op import SpatioTemporalGraph, sample_bernoulli_avialability_model
from schedule_generation import sample_occupancy, persistence_prob, generate_schedule


### Path planning for sampling based methods
def sample_best_path(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, budget, time_interval, num_intervals, curr_time, curr_node, maintenance_node,
        mu, maintenance_reward, deliver_reward, num_paths, num_worlds, incorporate_observation, incorporate_observation_hack, variance_bias, multiple_visits, replan, schedule_generation_method, use_gp):

    # generate potential solutions
    solution_set = []
    for path_index in range(num_paths):
        sample_availability_models = {}
        for request in requests_left_to_deliver:
            sample_availability_models[request] = sample_bernoulli_avialability_model(base_availability_models[request])
        
        st_g = SpatioTemporalGraph(sample_availability_models, base_model_variances, mu, int((budget - curr_time)/time_interval), budget - curr_time, time_interval, maintenance_node, maintenance_reward, deliver_reward, use_gp)
        st_g.build_graph(g, curr_node, curr_time, requests_left_to_deliver, availability_observations, incorporate_observation, incorporate_observation_hack, variance_bias)

        ## topological sort
        # print ("Sorting of the topological variety...")
        L = st_g.topological_sort()

        ## maximal profit path
        # print ("Calculating path...")
        # print ()
        path = st_g.calc_max_profit_path(L, requests_left_to_deliver, multiple_visits)

        solution_set.append(path)


    # generate evaluation worlds
    sim_worlds = {}
    for world_index in range(num_worlds):
        true_availability_models = {}
        for request in requests_left_to_deliver:
            true_availability_models[request] = sample_bernoulli_avialability_model(base_availability_models[request])
        sim_worlds[world_index] = generate_schedule(requests_left_to_deliver, true_availability_models, mu, num_intervals, schedule_generation_method, incorporate_observation)


    # evaluate potential solutions
    best_path = None
    best_score = -float("inf")
    for orig_path in solution_set:

        total_score = 0.0
        ave_score = 0.0
        for world_index in range(num_worlds):

            ### plan, execution loop
            num_requests = len(requests_left_to_deliver)
            sim_requests_left_to_deliver = copy.deepcopy(requests_left_to_deliver)
            sim_availability_observations = copy.deepcopy(availability_observations)
            total_profit = 0.0
            total_maintenance_profit = 0.0
            delivery_history = []
            sim_time = curr_time
            sim_curr_node = curr_node
            path_length = 1
            path_visits = 0

            orig_path_used = False


            while (path_visits < path_length):

                if orig_path_used:
                    st_g = SpatioTemporalGraph(base_availability_models, base_model_variances, mu, int((budget - sim_time)/time_interval), budget - sim_time, time_interval, maintenance_node, maintenance_reward, deliver_reward, use_gp)
                    st_g.build_graph(g, sim_curr_node, sim_time, sim_requests_left_to_deliver, sim_availability_observations, incorporate_observation, incorporate_observation_hack, variance_bias)

                    ## topological sort
                    # print ("Sorting of the topological variety...")
                    L = st_g.topological_sort()

                    ## maximal profit path
                    # print ("Calculating path...")
                    # print ()
                    path = st_g.calc_max_profit_path(L, sim_requests_left_to_deliver, multiple_visits)

                else:
                    path = orig_path
                    orig_path_used = True


                ### Execute
                path_visits = 1
                for visit in path[1:]:
                    if sim_curr_node == visit:
                        dist = 1
                    else:
                        # dist = g.get_distance(sim_curr_node, visit)
                        dist = g[sim_curr_node][visit]['weight']
                    sim_time += dist
                    sim_curr_node = visit
                    path_visits += 1
                    if visit == maintenance_node:
                        total_maintenance_profit += maintenance_reward
                    if visit in sim_requests_left_to_deliver:
                        # profit = availability_models[trial][visit](sim_time)
                        # total_profit += profit
                        # delivery_history.append([visit, sim_time])
                        # sim_requests_left_to_deliver.remove(visit)
                        curr_time_index = int(sim_time/time_interval)

                        # if curr_time_index > (num_intervals - 1):
                        #     print("Curr time index exceeds num intervals: " + str(curr_time_index) + ", " + str(num_intervals))
                        #     curr_time_index = num_intervals-1
                        assert(curr_time_index <= (num_intervals - 1))
                        available = sim_worlds[world_index][request][curr_time_index]
                        if bool(available):
                            sim_requests_left_to_deliver.remove(visit)
                            total_profit += deliver_reward
                            delivery_history.append([visit, sim_time])

                            if multiple_visits:
                                if replan:
                                    if incorporate_observation:
                                        sim_availability_observations[visit] = [1, sim_time]
                                    path_visits = 0
                                    path_length = 1
                                    break

                        else:
                            if replan:
                                # print ("Replanning, curr time: " + str(sim_time))
                                if incorporate_observation:
                                    sim_availability_observations[visit] = [0, sim_time]
                                path_visits = 0
                                path_length = 1
                                break

            ratio_divisor = num_requests + (budget - num_requests)*maintenance_reward
            competitive_ratio = (float(total_profit) + total_maintenance_profit)/ratio_divisor
            total_score += competitive_ratio

        ave_score = total_score/num_worlds
        if ave_score > best_score:
            best_path = orig_path
            best_score = ave_score


    return best_path



### Path visualization function specific to willow scenario
def visualize_path_willow(strategies, paths, availabilities, schedules, node_requests, maintenance_node, start_time, budget, time_interval, out_img):
   
    # time axis 
    X = np.array(list(range(int((budget-start_time)/time_interval))))
    X = time_interval*X
    X = X + start_time
    num_intervals = len(X)

    node_requests
    requested_nodes = []
    for i in range(44):
        index = "R"+str(i)
        if index in node_requests:
            requested_nodes.append(index)

    node_names = []
    node_names.append(maintenance_node)
    for request in requested_nodes:
        node_names.append(request)
    node_names.append("Other")
    num_node_indices = len(node_names)

    # true availability models
    availability_array = []
    availability_array.append([0.0 for i in range(num_intervals)])
    for request in requested_nodes:
        availability_array.append(availabilities[request])
    availability_array.append([0.0 for i in range(num_intervals)])
    availability_array = np.array(availability_array)

    # true schedules
    schedule_array = []
    schedule_array.append([0.0 for i in range(num_intervals)])
    for request in requested_nodes:
        schedule_array.append(schedules[request])
    schedule_array.append([0.0 for i in range(num_intervals)])
    schedule_array = np.array(schedule_array)

    # visits
    visit_traces = {}
    for strategy in strategies:
        visit_trace = []
        for visit in paths[strategy]:
            if visit in node_names:
                visit_trace.append(node_names.index(visit))
            else:
                visit_trace.append(num_node_indices-1)
        visit_traces[strategy] = visit_trace


    
    num_strategies = len(strategies)
    subplot_num1 = int(math.ceil(num_strategies/2))
    subplot_num2 = 2
    if num_strategies == 1:
        subplot_num2 = 1

    # subplot_num1 = 1
    # subplot_num2 = 1


    ## plot availability
    # fig, ax = plt.subplots()
    fig = plt.figure()
    subplot_index = 0
    for strategy in strategies:
        subplot_index += 1

        ax = plt.subplot(subplot_num1,subplot_num2,subplot_index)
        plt.title(strategy)

        im = ax.imshow(availability_array)
        # im = ax.imshow(schedule_array)

        # ax.set_xticks(np.arange(len(X)))
        # ax.set_yticks(np.arange(len(node_names)))
        # ax.set_xticklabels(X)
        # ax.set_yticklabels(node_names)
        # plt.xticks(range(min(X), max(X)+1, 1))
        # ax.get_yaxis().set_visible(False)
        ax.axes.get_yaxis().set_ticks([])

        plt.xlabel("Time (min)")
        plt.ylabel("Nodes")

        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")

        time_index = 0
        for visit in visit_traces[strategy]:
            text = ax.text(time_index, visit, "x",
                           ha="center", va="center", color="w")
            time_index += 1

        # ax.set_title("Path Trace vs Probs")
        ax.set_title(strategy)
        fig.tight_layout()

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Availability", rotation=-90, va="bottom")

    plt.suptitle("Path Trace vs Node Availability Probabilities")
    plt.savefig(out_img + "_avail.jpg")
    # plt.show()





    ## plot schedule
    # fig, ax = plt.subplots()
    fig = plt.figure()
    subplot_index = 0
    for strategy in strategies:
        subplot_index += 1

        ax = plt.subplot(subplot_num1,subplot_num2,subplot_index)
        plt.title(strategy)

        # im = ax.imshow(availability_array)
        im = ax.imshow(schedule_array)

        # ax.set_xticks(np.arange(len(X)))
        # ax.set_yticks(np.arange(len(node_names)))
        # ax.set_xticklabels(X)
        # ax.set_yticklabels(node_names)
        # plt.xticks(range(min(X), max(X)+1, 1))
        # ax.get_yaxis().set_visible(False)
        ax.axes.get_yaxis().set_ticks([])

        plt.xlabel("Time (min)")
        plt.ylabel("Nodes")

        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")

        time_index = 0
        for visit in visit_traces[strategy]:
            text = ax.text(time_index, visit, "x",
                           ha="center", va="center", color="w")
            time_index += 1

        # ax.set_title("Path Trace vs Schedule")
        ax.set_title(strategy)
        fig.tight_layout()

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Availability", rotation=-90, va="bottom")

    plt.suptitle("Path Trace vs Node Schedules")
    plt.savefig(out_img + "_schedule.jpg")
    # plt.show()





### Template for various path planner implementations
def plan_path(strategy, g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    if strategy == 'no_temp':
        return plan_path_no_temp_info(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'no_replan':
        return plan_path_no_observe(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'replan_no_observe':
        return plan_path_no_observe(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'hack_observe':
        return plan_path_with_hack_observe(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'observe':
        return plan_path_with_observe(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'observe_mult_visits':
        return plan_path_with_observe_mult_visits(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'observe_sampling':
        return plan_path_with_observe_sampling(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'observe_sampling_variance_bias':
        return plan_path_replan_with_observe_sampling_variance_bias(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    elif strategy == 'observe_sampling_mult_visits':
        return plan_path_replan_with_observe_sampling_mult_visits(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params)
    else:
        raise ValueError(strategy)


def plan_path_no_temp_info(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    constant_availability_models = {}
    constant_model_variances = {}
    for request in requests_left_to_deliver:
        avails = []
        variances = []
        num_intervals = int(params['budget']/params['time_interval'])
        # for i in range(len(base_availability_models[request])):
        for i in range(num_intervals):
            avails.append(1.0)
            variances.append(0.0)
        constant_availability_models[request] = avails
        constant_availability_models[request] = variances

    st_g = SpatioTemporalGraph(constant_availability_models, constant_model_variances, mu, int((params['budget'] - curr_time)/params['time_interval']), params['budget'] - curr_time, params['time_interval'], params['maintenance_node'], params['maintenance_reward'], params['deliver_reward'], False)
    st_g.build_graph(g, curr_node, curr_time, requests_left_to_deliver, availability_observations, False, False, False)
    L = st_g.topological_sort()
    path = st_g.calc_max_profit_path(L, requests_left_to_deliver, False)
    return path

def plan_path_no_observe(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    st_g = SpatioTemporalGraph(base_availability_models, base_model_variances, mu, int((params['budget'] - curr_time)/params['time_interval']), params['budget'] - curr_time, params['time_interval'], params['maintenance_node'], params['maintenance_reward'], params['deliver_reward'], params['use_gp'])
    st_g.build_graph(g, curr_node, curr_time, requests_left_to_deliver, availability_observations, False, False, False)
    L = st_g.topological_sort()
    path = st_g.calc_max_profit_path(L, requests_left_to_deliver, False)
    return path

def plan_path_with_hack_observe(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    st_g = SpatioTemporalGraph(base_availability_models, base_model_variances, mu, int((params['budget'] - curr_time)/params['time_interval']), params['budget'] - curr_time, params['time_interval'], params['maintenance_node'], params['maintenance_reward'], params['deliver_reward'], params['use_gp'])
    st_g.build_graph(g, curr_node, curr_time, requests_left_to_deliver, availability_observations, True, True, False)
    L = st_g.topological_sort()
    path = st_g.calc_max_profit_path(L, requests_left_to_deliver, False)
    return path

def plan_path_with_observe(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    st_g = SpatioTemporalGraph(base_availability_models, base_model_variances, mu, int((params['budget'] - curr_time)/params['time_interval']), params['budget'] - curr_time, params['time_interval'], params['maintenance_node'], params['maintenance_reward'], params['deliver_reward'], params['use_gp'])
    st_g.build_graph(g, curr_node, curr_time, requests_left_to_deliver, availability_observations, True, False, False)
    L = st_g.topological_sort()
    path = st_g.calc_max_profit_path(L, requests_left_to_deliver, False)
    return path

def plan_path_with_observe_mult_visits(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    st_g = SpatioTemporalGraph(base_availability_models, base_model_variances, mu, int((params['budget'] - curr_time)/params['time_interval']), params['budget'] - curr_time, params['time_interval'], params['maintenance_node'], params['maintenance_reward'], params['deliver_reward'], params['use_gp'])
    st_g.build_graph(g, curr_node, curr_time, requests_left_to_deliver, availability_observations, True, False, False)
    L = st_g.topological_sort()
    path = st_g.calc_max_profit_path(L, requests_left_to_deliver, True)
    return path

def plan_path_with_observe_sampling(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    path = sample_best_path(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, params['budget'], params['time_interval'], params['num_intervals'], curr_time, curr_node, params['maintenance_node'],
            mu, params['maintenance_reward'], params['deliver_reward'], params['num_paths'], params['num_worlds'], True, False, 0.0, False, True, params['schedule_generation_method'], params['use_gp'])
    return path

def plan_path_replan_with_observe_sampling_variance_bias(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    path = sample_best_path(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, params['budget'], params['time_interval'], params['num_intervals'], curr_time, curr_node, params['maintenance_node'],
            mu, params['maintenance_reward'], params['deliver_reward'], params['num_paths'], params['num_worlds'], True, False, params['variance_bias'], False, True, params['schedule_generation_method'], params['use_gp'])
    return path

def plan_path_replan_with_observe_sampling_mult_visits(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, curr_time, curr_node, mu, params):
    path = sample_best_path(g, base_availability_models, base_model_variances, availability_observations, requests_left_to_deliver, params['budget'], params['time_interval'], params['num_intervals'], curr_time, curr_node, params['maintenance_node'],
            mu, params['maintenance_reward'], params['deliver_reward'], params['num_paths'], params['num_worlds'], True, False, 0.0, True, True, params['schedule_generation_method'], params['use_gp'])
    return path
