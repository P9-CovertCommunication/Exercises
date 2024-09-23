# Write a code simulating a behavior of a system consisting of 10 servers and having Blocked
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

    data_dict = {}
    
    for Customers in [100, 10000, 100000]:   
        sum_percent_blocked = 0
        sum_percent_blocked_time = 0
        for arrival_process in ["M", "C"]:
            for service_process in ["M", "C"]:
                num_simulations = int(10**6/Customers)
                blocked_List = []
                blocked_time_list = []
                for _ in tqdm(range(num_simulations)):     
                    blocked, blocked_time = BCC_simulator(arrival_process, service_process, int(Customers))
                    blocked_List.append(blocked)
                    blocked_time_list.append(blocked_time)
                print(f"Number of customers: {Customers}, Queue: {arrival_process}/ {service_process} /10/0")
                print("Average Percentage of blocked calls: ", np.mean(blocked_List))
                print("sample variance of blocked calls: ", np.var(blocked_List))
                print("Average Percentage of time in blocked state: ", np.mean(blocked_time_list))
                print("sample variance of blocked time: ", np.var(blocked_time_list))

                print("\n")
                
                key = (Customers, arrival_process, service_process)
                data_dict[key] = {
                    "Blocked Percent": avg_blocked_calls,
                    "Variance blocked percent": var_blocked_calls,
                    "Blocked Time": avg_blocked_time,
                    "Variance blocked time": var_blocked_time
                   
                }
    
    #Save dict as pickle
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    print("Simulation completed")
                

