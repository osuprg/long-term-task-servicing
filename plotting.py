import numpy as np
import matplotlib.pyplot as plt
import csv
import math
# import seaborn as sns
# import pandas as pd

def generate_plots(strategies, num_deliveries_runs, availability_percents, csv_filename, plotting_mode='cr'):
    maintenance_crs = {}
    competitive_ratios = {}
    for num_deliveries in num_deliveries_runs:
        maintenance_crs[str(num_deliveries)] = {}
        competitive_ratios[str(num_deliveries)] = {}
        for strategy in strategies:
                maintenance_crs[str(num_deliveries)][strategy] = {}
                competitive_ratios[str(num_deliveries)][strategy] = {}
                for availability_percent in availability_percents:
                    maintenance_crs[str(num_deliveries)][strategy][str(availability_percent)] = []
                    competitive_ratios[str(num_deliveries)][strategy][str(availability_percent)] = []
            


    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            strategy = row[0]
            # budget = int(row[1])
            num_deliveries = int(row[2])
            assert (num_deliveries in num_deliveries_runs)
            availability_percent = float(row[3])
            # availability_chance = float(row[4])
            maintenance_reward = float(row[5])
            noise_amp = float(row[6])
            noise_amp = float(row[7])
            competitive_ratio = float(row[8])
            maintenance_cr = float(row[9])
            

            if num_deliveries in num_deliveries_runs:
                mcrs = maintenance_crs[str(num_deliveries)][strategy][str(availability_percent)]
                mcrs.append(maintenance_cr)
                maintenance_crs[str(num_deliveries)][strategy][str(availability_percent)] = mcrs

                crs = competitive_ratios[str(num_deliveries)][strategy][str(availability_percent)]
                crs.append(competitive_ratio)
                competitive_ratios[str(num_deliveries)][strategy][str(availability_percent)] = crs

        
        # calc stats
        maintenance_cr_stats = {}
        competitive_ratio_stats = {}
        for num_deliveries in num_deliveries_runs:
            maintenance_cr_stats[str(num_deliveries)] = {}
            competitive_ratio_stats[str(num_deliveries)] = {}
            for strategy in strategies:
                    maintenance_cr_stats[str(num_deliveries)][strategy] = {}
                    competitive_ratio_stats[str(num_deliveries)][strategy] = {}
                    for availability_percent in availability_percents:

                        competitive_ratio_ave = np.mean(np.array(competitive_ratios[str(num_deliveries)][strategy][str(availability_percent)]))
                        competitive_ratio_stdev = np.std(np.array(competitive_ratios[str(num_deliveries)][strategy][str(availability_percent)]))
                        maintenance_cr_ave = np.mean(np.array(maintenance_crs[str(num_deliveries)][strategy][str(availability_percent)]))
                        maintenance_cr_stdev = np.std(np.array(maintenance_crs[str(num_deliveries)][strategy][str(availability_percent)]))

                        maintenance_cr_stats[str(num_deliveries)][strategy][str(availability_percent)] = [maintenance_cr_ave, maintenance_cr_stdev]
                        competitive_ratio_stats[str(num_deliveries)][strategy][str(availability_percent)] = [competitive_ratio_ave, competitive_ratio_stdev]

    
        fig = plt.figure()
        subplot_num1 = int(math.ceil(len(num_deliveries_runs)/2))
        subplot_num2 = 2
        if len(num_deliveries_runs) == 1:
            subplot_num2 = 1
        index = 0
        for num_deliveries in num_deliveries_runs:
            index += 1

            if plotting_mode == 'mr':
                plt.subplot(subplot_num1,subplot_num2,index)
                plt.title("Num Deliveries: " + str(num_deliveries))
                # plt.xlabel("Availability Percent")
                # plt.ylabel("Last Deliver Time")
                # plt.legend()

            if plotting_mode == 'cr':
                plt.subplot(subplot_num1,subplot_num2,index)
                plt.title("Num Deliveries: " + str(num_deliveries))
                # plt.xlabel("Availability Percent")
                # plt.ylabel("Competitive Ratio")
                # plt.legend()

            for strategy in strategies:
                mr_y = []
                mr_y_sd = []
                cr_y = []
                cr_y_sd = []
                for availability_percent in availability_percents:
                    mr_y.append(maintenance_cr_stats[str(num_deliveries)][strategy][str(availability_percent)][0])
                    mr_y_sd.append(maintenance_cr_stats[str(num_deliveries)][strategy][str(availability_percent)][1])
                    cr_y.append(competitive_ratio_stats[str(num_deliveries)][strategy][str(availability_percent)][0])
                    cr_y_sd.append(competitive_ratio_stats[str(num_deliveries)][strategy][str(availability_percent)][1])

                x = np.array(availability_percents)
                mr_y = np.array(mr_y)
                mr_y_sd = np.array(mr_y_sd)
                cr_y = np.array(cr_y)
                cr_y_sd = np.array(cr_y_sd)

                if plotting_mode == 'mr':
                    # data = {}
                    # data['x'] = x
                    # data['mr_y'] = mr_y
                    # data['mr_y_sd'] = mr_y_sd
                    # df = pd.DataFrame.from_dict(data)
                    # sns.lineplot(x='x', data=df)

                    plt.subplot(subplot_num1,subplot_num2,index)
                    # plt.errorbar(x, mr_y, mr_y_sd, label=strategy)
                    plt.plot(x, mr_y, label=strategy)
                    plt.fill_between(x, mr_y-mr_y_sd, mr_y+mr_y_sd, alpha=0.2)


                if plotting_mode == 'cr':
                    # data = {}
                    # data['x'] = x
                    # data['cr_y'] = cr_y
                    # data['cr_y_sd'] = cr_y_sd
                    # df = pd.DataFrame.from_dict(data)

                    ax = plt.subplot(subplot_num1,subplot_num2,index)
                    ax.set_ylim(0.0, 1.0)
                    # plt.errorbar(x, cr_y, cr_y_sd, label=strategy)
                    plt.plot(x, cr_y, label=strategy)
                    plt.fill_between(x, cr_y-cr_y_sd, cr_y+cr_y_sd, alpha=0.2)


        if plotting_mode == 'mr':
            plt.suptitle("Maintenance CR by Availability Percent for Num Del: " + str(num_deliveries))  
        if plotting_mode == 'cr':
            plt.suptitle("Competitive Ratio by Availability Percent for Num Del: " + str(num_deliveries))           
        plt.legend(loc='lower right')
        plt.show()





