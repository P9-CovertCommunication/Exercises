
# I want to plot the queue size vs time for all the users. The format is 
    # Time, Queue size
    # 1732267913.391865,56
    # 1732267913.4071562,62
    # 1732267913.4214168,65
    # 1732267913.4358552,73
    # 1732267913.4495077,80
    # 1732267913.464416,87
    # 1732267913.4803007,93

# I will use the following code to plot the data for all the TCP types

import os
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

TCP_types = ["cubic", "reno", "vegas"]

for i in range(2):
    for TCP_type in TCP_types:
        for n in range(1,10):

            path = f"Mini_Project/Part4/data_22_11/LOSS{i}_{TCP_type}/{n}/"
            files = glob.glob(path + "R1_qlen.txt")
            for file in files:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    time = []
                    queue = []
                    for line in lines:
                        parts = line.split(',')
                        time.append(float(parts[0]))
                        queue.append(float(parts[1]))
                    time = [t - time[0] for t in time]  # Subtract the first time
                    plt.plot(time, queue, label=f"{TCP_type} with {n} users")
            plt.xlabel("Time")
            plt.ylabel("Queue size")
            plt.title(f"Queue size vs Time for {TCP_type} with {n} users")
            plt.legend()
            plt.savefig(f"Mini_Project/Part4/data_22_11/Queue_plots/QUEUE_LOSS{i}_{TCP_type}_n{n}.png")
            plt.close()
            print(f"Plot saved as {path}Queue_plot.png")
