import sys
import yaml
import os.path
from test_runs import vizualize_sample_execution



### High level code for visualizing sample execution
def main(stat_run, input_file):

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
    visualize = bool(int(inputs['visualize']))
    out_gif_path = filepath + inputs['visualize_path']
    out_img_path = filepath + inputs['out_img_path']

    # planning and execution
    vizualize_sample_execution(world_config_file, schedule_config_file, planner_config_file, model_config_file, base_model_filepath, schedule_filepath, strategies, num_deliveries, availability_percents, stat_run, visualize, out_gif_path, out_img_path)


if __name__ == "__main__":
    input_file = sys.argv[1]
    if len(sys.argv) == 2:
        main(0, input_file)
    else:
        print("Usage: <input file>")
