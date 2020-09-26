import sys
import yaml
import os.path
from test_runs import stat_runs



### High level code for running stat runs of task planning and simulated execution
def main(num_stat_runs, input_file, output_file):

    filepath = os.path.dirname(os.path.abspath(__file__))
    with open(filepath + input_file) as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)
    world_config_file = filepath + inputs['world_config_file']
    schedule_config_file = filepath + inputs['schedule_config_file']
    planner_config_file = filepath + inputs['planner_config_file']
    model_config_file = filepath + inputs['model_config_file']
    base_model_filepath = filepath + inputs['base_model_filepath']
    schedule_filepath = filepath + inputs['schedule_filepath']
    strategies = inputs['strategies']
    num_deliveries = []
    for i in inputs['num_deliveries']:
        num_deliveries.append(int(i))
    availability_percents = []
    for i in inputs['availability_percents']:
        availability_percents.append(float(i))
    budgets = []
    for i in inputs['budgets']:
        budgets.append(int(i))
    if not(output_file is None):
        output_file = filepath + output_file
    visualize = bool(int(inputs['visualize']))
    out_gif_path = filepath + inputs['visualize_path']

    # planning and execution
    stat_runs(world_config_file, schedule_config_file, planner_config_file, model_config_file, base_model_filepath, schedule_filepath, output_file, strategies, num_deliveries, availability_percents, budgets, num_stat_runs, visualize, out_gif_path)


if __name__ == "__main__":
    num_stat_runs = int(sys.argv[1])
    input_file = sys.argv[2]
    if len(sys.argv) == 3:
        main(num_stat_runs, input_file, None)
    elif len(sys.argv) == 4:
        output_file = sys.argv[3]
        main(num_stat_runs, input_file, output_file)
    else:
        print("Usage: <num stat runs> <input file> <output file=None>")
