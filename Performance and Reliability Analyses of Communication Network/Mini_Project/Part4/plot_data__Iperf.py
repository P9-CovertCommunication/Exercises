# I Want to plot the data in the filepath 
# Mini_Project\Part4\LOSS1_[TCP_TYPE]\[Number_Of_Users]\S[Serve(S) number].txt.
# The file have the following format:
# ------------------------------------------------------------
# Server listening on TCP port 5001
# TCP window size: 85.3 KByte (default)
# ------------------------------------------------------------
# [  6] local 10.0.0.2 port 5001 connected with 10.0.0.1 port 56486
# [ ID] Interval       Transfer     Bandwidth
# [  6]  0.0- 1.0 sec  1.11 MBytes  9.34 Mbits/sec
# [  6]  2.0- 3.0 sec  90.5 KBytes   741 Kbits/sec
# I want to calculate a avg Bandwidth of alle the users, and plot the avg Bandwidth vs the number of serveres,
# The number of Servers is indicated by the number of folders in the path Mini_Project\Part4\LOSS1_cubic\[Number_Of_Users] 
# with the plot should also show the varience of the data

# I will use the following code to plot the data for all the TCP types

import os
import matplotlib.pyplot as plt

TCP_types = ["cubic", "reno", "vegas"]
for TCP_type in TCP_types:
    # Path to the data
    path = "Mini_Project//Part4//LOSS1_{0}".format(TCP_type)
    # Get all the folders in the path
    folders = os.listdir(path)
    # Get the number of folders
    num_folders = len(folders)
    # Create a list to store the average bandwidth
    avg_bandwidth = []
    # Create a list to store the varience of the data
    varience = []
    # Loop through the folders
    for folder in folders:
        # Get the path to the folder
        folder_path = os.path.join(path, folder)
        # Get all the files in the folder
        files = os.listdir(folder_path)
        # Create a list to store the bandwidth
        bandwidth = []
        # Loop through the files
        for file in files:
            # Get the path to the file
            file_path = os.path.join(folder_path, file)
            # Open the file
            with open(file_path, 'r') as f:
                # Read the lines
                lines = f.readlines()
                # Loop through the lines
                for line in lines:
                    # Check if the line contains the bandwidth IN "Mbits/sec" OR "kbits/sec"
                    if "Mbits/sec" in line:
                        # Split the line
                        parts = line.split()
                        # Get the bandwidth
                        bandwidth.append(float(parts[-2]))
                    elif "kbits/sec" in line:
                        # Split the line
                        parts = line.split()
                        # Get the bandwidth
                        bandwidth.append(float(parts[-2])/1000)
        # Calculate the average bandwidth
        avg_bandwidth.append(sum(bandwidth)/len(bandwidth))
        # Calculate the varience
        varience.append(sum([(x - sum(bandwidth)/len(bandwidth))**2 for x in bandwidth])/len(bandwidth))
    # Plot the data as a bar chart, with the different TCP_types next to each other, with error bars
    bar_width = 0.25
    index = range(1, num_folders + 1)
    plt.bar([i + bar_width * TCP_types.index(TCP_type) for i in index], avg_bandwidth, bar_width, yerr=varience, label="TCP type: {0}".format(TCP_type))
plt.xlabel("Number of Servers")
plt.ylabel("Bandwidth Mbits/sec")
plt.xticks([i + bar_width for i in index], index)
plt.legend()
plt.show()

    
    
    
#     plt.plot(range(1, num_folders+1), avg_bandwidth, label="Average Bandwidth, TCP type: {0}".format(TCP_type))
#     plt.errorbar(range(1, num_folders+1), avg_bandwidth, yerr=varience, label="Varience, TCP type: {0}".format(TCP_type))
# plt.xlabel("Number of Servers")
# plt.ylabel("Bandwidth")
# plt.legend()
# plt.show()



