#################################################################################################
#                                                                                               #
# This is the property of the Authors, we gladly accept donations in the form of beer.          #
# Authors: Anders Bundgaard and Nicolai Lyholm                                                  #                                                   
# Date: 23/9/2024                                                                               #      
#                                                                                               #
#################################################################################################

# 
# 
#  Write a code simulating a behavior of a system consisting of 10 servers and having Blocked
# Call Cleared (BCC) behavior.
# Consider 4 different cases:
# 1. Poisson arrivals, exponential service times.
# 2. Poisson arrivals, constant service times.
# 3. Constant interarrival times, exponential service times.
# 4. Constant interarrival times, constant service times.
# The arrival rate is set to 4 and the average service time 2.4 (so that the offered load is 4 x
# 2.4 = 9.6 erlangs).
# The simulations should be run for

# a) 100 arriving customers;\\
# b) 10.000 arriving customers;\\
# c) 100.000 arriving customers.\\

# Calculate probability of blocking given by Erlang B formula; P10 and 𝜋10 . The results should
# be filled in the following table:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import math
RUN_SIM = False

def erlang_b(servers, offered_load):
    offered_load = offered_load
    servers = servers
    erlang_b = ((offered_load**servers)/math.factorial(servers))/np.sum([(offered_load**i)/math.factorial(i) for i in range(servers+1)])
    return erlang_b

def BCC_simulator(arrival : str, service : str, customers : int): 
    arrival_rate = 4
    service_rate = 1/2.4
    servers = 10
    customers=customers
    
    if arrival == "M":
        arrival_times = np.cumsum(np.random.exponential(1/arrival_rate, customers)) #Poission arrival
    else:
        arrival_times = np.arange(0,customers*1/arrival_rate, 1/arrival_rate)
    
    if service == "M":
        service_time = np.random.exponential(1/service_rate, customers)

    else:
        service_time = np.ones(customers) * 1/service_rate
        
        
    # Initialize the task end times of the servers
    end_times = np.zeros(servers)
    time_in_blocked_state = 0
    num_blocked = 0

    # print(arrival_times)
    # print(service_time)
    for i in range(customers):
        # Find the server that will be available first
        server = np.argmin(end_times)
        
        # If no server is available, block the call
        if end_times[server] <= arrival_times[i]:
            end_times[server] = arrival_times[i] + service_time[i]

        # Log if the call is blocked and for how long 
            server = np.argmin(end_times)
            if end_times[server] > arrival_times[i]:
                time_in_blocked_state = time_in_blocked_state + end_times[server] - arrival_times[i]

        elif end_times[server] > arrival_times[i]:
            # print("Call blocked")
            num_blocked = num_blocked + 1
            
    percent_blocked = num_blocked/customers
    percent_blocked_time = time_in_blocked_state/np.max(end_times)
    # print("PI, Percentage of blocked calls: ", percent_blocked)
    # print("P, Percentage of time in blocked state: ",percent_blocked_time)
    return percent_blocked, percent_blocked_time

if __name__ == "__main__":
    if RUN_SIM:
        data_dict = {}
        
        for Customers in [100, 10000, 100000]:   
            sum_percent_blocked = 0
            sum_percent_blocked_time = 0
            for arrival_process in ["M", "C"]:
                for service_process in ["M", "C"]:
                    num_simulations = int(10**7/Customers)
                    blocked_List = []
                    blocked_time_list = []
                    for _ in tqdm(range(num_simulations)):     
                        blocked, blocked_time = BCC_simulator(arrival_process, service_process, int(Customers))
                        blocked_List.append(blocked)
                        blocked_time_list.append(blocked_time)
                        
                    print(f"Number of customers: {Customers}, Queue: {arrival_process}/ {service_process} /10/0")
                    print(f"Average Percentage of blocked calls: {np.mean(blocked_List)*100} [%]")
                    print(f"sample variance of blocked calls: {np.var(blocked_List)*100} [%]")
                    print(f"Average Percentage of time in blocked state: {np.mean(blocked_time_list)*100}[%]")
                    print(f"sample variance of blocked time: {np.var(blocked_time_list)*100} [%]")

                    print("\n")
                    
                    key = (Customers, arrival_process, service_process)
                    data_dict[key] = {
                        "Mean blocked [%]": np.mean(blocked_List)*100,
                        "Variance blocked [%]": np.var(blocked_List)*100,
                        "Mean blocked Time [s]": np.mean(blocked_time_list)*100,
                        "Variance blocked time [s]": np.var(blocked_time_list)*100
                    }
            #Save dict as pickle
        with open('saved_dictionary.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        print("Simulation completed")
    
    else:
        # print(erlang_b(10, 9.6)*100)
        with open('saved_dictionary.pkl', 'rb') as f:
            data = pickle.load(f)

        for customers in [100, 10000, 100000]:
            print(f"========================Customers {customers}========================")
            print("\n")
            for arrival_process in ["M", "C"]:
                    for service_process in ["M", "C"]:
                        key = (customers, arrival_process, service_process)
                        print(f"========RESULTS {arrival_process}/{service_process}/10/0========")
                        print(f"Simulated mean blocking: {data[key]["Mean blocked [%]"]:.5f} % | Var {data[key]["Variance blocked [%]"]:.5f} %")
                        print(f"Simulated mean blocking time: {data[key]["Mean blocked Time [s]"]:.5f} s | Var {data[key]["Variance blocked time [s]"]:.5f} s")
                        
                        print(f"Erlang B blocking: {erlang_b(10, 9.6)*100:.5f}")    
                        print("\n")
            print("\n")
            print("\n")
        print(f"To conclude:\nIt can be seen that the Erlang B blocking probability is very close to the simulated blocking probability when the Arrical process is Memoryless/poisson")
        print(f"It can also be seen that the variance of the blocking time fals when the number of customers increases, i.e. if one does not take a mean over many simulations the no. customers must be high")   

                
