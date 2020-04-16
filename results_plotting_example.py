import sys
import yaml
import os.path
from test_runs import stat_runs
from plotting import generate_plots


### High level code for generating results plots
def main(input_file, results_file):

    filepath = os.path.dirname(os.path.abspath(__file__))
    with open(filepath + input_file) as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)
    world_config_file = filepath + inputs['world_config_file']
    schedule_config_file = filepath + inputs['schedule_config_file']
    planner_config_file = filepath + inputs['planner_config_file']
    base_model_filepath = filepath + inputs['base_model_filepath']
    schedule_filepath = filepath + inputs['schedule_filepath']
    strategies = inputs['strategies']
    num_deliveries = []
    for i in inputs['num_deliveries']:
        num_deliveries.append(int(i))
    availability_percents = []
    for i in inputs['availability_percents']:
        availability_percents.append(float(i))

    # plotting
    generate_plots(strategies, num_deliveries, availability_percents, results_file, plotting_mode='cr')

if __name__ == "__main__":
    input_file = sys.argv[1]
    results_file = sys.argv[2]
    if len(sys.argv) == 3:
        main(input_file, results_file)
    else:
        print("Usage: <input file> <results file>")
