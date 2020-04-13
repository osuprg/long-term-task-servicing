import yaml
import os
from test_runs import stat_runs

# willow, full day, 1hr windows
# params
# strategies: 

filepath = os.path.dirname(os.path.abspath(__file__))
param_config_file = filepath + "/config/params/params.yaml"
scenario_config_file = filepath + "/config/scenarios/willow_scenario.yaml"
base_model_filepath = filepath + "/worlds/models/params_willow_scenario_1/"
schedule_filepath = filepath + "/worlds/schedules/params_willow_scenario_1/"
save_csv = filepath + "/output/out.csv"
out_img = filepath + "/figs/out.jpg"

# strategies = ['no_temp', 'no_replan', 'replan_no_observe', 'hack_observe', 'observe', 'observe_sampling', 'observe_sampling_variance_bias', 'observe_sampling_mult_visits']
strategies = 'no_temp', 'no_replan', 'replan_no_observe', 'hack_observe', 'observe', 'observe_sampling', 'observe_sampling_variance_bias', 'observe_sampling_mult_visits'

# num_deliveries_runs = [10, 20, 30, 40]
num_deliveries_runs = [10]

# availability_percents = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
availability_percents = [.2]

num_stat_runs = 1
record_output = False
generate_schedules = False

# planning and execution
stat_runs(param_config_file, scenario_config_file, base_model_filepath, schedule_filepath, save_csv, out_img, strategies, num_deliveries_runs, availability_percents, num_stat_runs, record_output, generate_schedules)


# plotting
# generate_plots(strategies, num_deliveries_runs, availability_percents, save_csv, plotting_mode='cr')
