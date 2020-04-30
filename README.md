# long-term-task-servicing
Project for modeling probabilistic, temporal availability of service tasks and exploiting models for efficient service. Currently being investigated in the context of a package delivery scenario for an autonomous delivery service robot.

# example_tests.py
High level file for running stat runs of planning and simulated execution. Run with: python3 example_tests.py <num_stat_runs> <input_file> <output_file>. input_file should follow format similar to those in experiment_files/input/ and will notably include relative locations of config files for world generation, schedule generation, and planner parameters. Additionally will include which planner strategies to test along with numbers of scheduled deliveries and general schedule availabilities to test against. output_file is optional; if included will output a csv file of results to be plotted using results_plotting_example.py. num_stat_runs specifies number of experiments to run for each planner, test combination.

Example: python3 example_tests.py 5 /experiment_files/input/willow_8hrs_no_sampling.yaml /experiment_files/output/willow_8hrs_no_sampling.csv

# results_plotting_example.py
High level file for plotting results generated from running example_tests.py. Run with: python3 results_plotting_example.py <input_file> <results_file>, where input_file is the associated input_file given to example_tests.py and results_file is the output_file it produces. Competitive ratio (collected reward from task servicing over conservative estimate of total possible reward) will be plotted for strategies specified in input_file for specified number of deliveries and schedule availabilities.

Example: python3 results_plotting_example.py /experiment_files/input/willow_8hrs_no_sampling.yaml /experiment_files/output/willow_8hrs_no_sampling.csv

# visualization_example.py
High level file for visualizing sample execution. Produces timeline graphic with nodes and corresponding schedules/availabilities overlaid with indicator designating nodes visited during execution. Run with: python3 visualization_example.py <input_file> where input_file is the same as would be given to example_tests.py. For each planning strategy, test scenario combination included in input_file, visited nodes will be overlayed against true availabilities and schedules of associated nodes. If visualization is set to 1 in the input file, gif animation of execution will additionally be generated in experimental_files/output/animaations.

Example: python3 visualization_example.py /experiment_files/input/willow_30min_no_sampling_execution_viz_example.yaml

# test_runs.py
High level functions for running planners on simulated worlds as well as visualizing sample execution.

# plan_execution_simulation.py
Evaluates execution simulation of planner applied to task delivery scenario.

# planners.py
Functions for constructed plans for various planning strategies. Additionally contains functions for producing path visualizations. Currently implemented planners include no_temp (no temporal info used), no_replan (no replanning around execution failure), replan_no_observe (replanning without updating temporal models around execution failure), hack_observe (simplistic update of temporal model with observations), observe (update temporal model accounting for temporal persistence when observations are encountered), observe_sampling (additionally simulate multiple possible exeuctions and plan variations, choose plan that performs best), observe_sampling_variance_bias (bias toward nodes with low variance on availability probability), observe_sampling_mult_visits (allow planning to consider multiple visits, spread out expected reward accordingly) 

# world_generation.py
Functions for creating simulated schedules for given graph representations. Base probability models to be used in planning/associated true models and schedules are generally saved and loaded for experiment repeatability.

# td_op.py
SpatioTemporalGraph class implementation of (Ma, et al., 2017)[1] with additional modifications along with supporting spatial graph class. build_graph creates spatio temporal representation of spatial nodes at every time slice within budget with edges connecting nodes at different time slices according to associated travel costs. topological_sort produces ordering to allow for efficient calculation of best path using calc_max_profit_path. Modifications include taking into account temporal persistence model of (Toris and Chernova, 2017)[2] as well as additional information needed for various planner strategies.

[1] Ma, Zhibei, et al. "A Spatio-Temporal Representation for the Orienteering Problem with Time-Varying Profits." IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2017.

[2] Toris, Russell, and Sonia Chernova. "Temporal Persistence Modeling for Object Search." IEEE International Conference on Robotics and Automation (ICRA). 2017.

# utils.py
Various supporting functions, notably for file I/O.

# plotting.py
Results plotting code

# config/
Sub-folder holding config files containing world generation, schedule generation, and planner parameters.

# worlds/
Sub-folder for saving generated models/schedules to allow for experiment repeatability.

# experiment_files/
Sub-folder for input and output files for high level stat runs.

