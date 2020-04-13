import sys
import yaml
import os
from test_runs import stat_runs

# willow, full day, 1hr windows
# params
# strategies: 


def main(input_file):

	filepath = os.path.dirname(os.path.abspath(__file__))


	with open(input_file) as f:
		inputs = yaml.load(f, Loader=yaml.FullLoader)
	world_config_file = filepath + inputs['world_config_file']
	schedule_config_file = filepath + inputs['schedule_config_file']
	planner_config_file = filepath + inputs['planner_config_file']
	base_model_filepath = filepath + inputs['base_model_filepath']
	schedule_filepath = filepath + inputs['schedule_filepath']
	inputs['strategies'] = strategies
	num_deliveries = []
	for i in range(len(inputs['num_deliveries'])):
		num_deliveries.append(int(i))
	inputs['num_deliveries'] = num_deliveries
	availability_percents = []
	for i in range(len(inputs['availability_percents'])):
		availability_percents.append(float(i))
	inputs['availability_percents'] = availability_percents

	save_csv = filepath + "/output/out.csv"
	out_img = filepath + "/figs/out.jpg"


	num_stat_runs = 1
	record_output = False
	generate_schedules = False

	# planning and execution
	stat_runs(world_config_file, schedule_config_file, planner_config_file, base_model_filepath, schedule_filepath, save_csv, out_img, strategies, num_deliveries_runs, availability_percents, num_stat_runs, record_output, generate_schedules)


	# plotting
	# generate_plots(strategies, num_deliveries_runs, availability_percents, save_csv, plotting_mode='cr')



if __name__ == "__main__":
	input_file = sys.argv[1]
    main(input_file)
