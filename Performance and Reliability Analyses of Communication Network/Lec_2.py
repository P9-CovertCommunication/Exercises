#################################################################################################
#                                                                                               #
# This is the property of the Authors, we gladly accept donations in the form of beer.          #
# Authors: Anders Bundgaard and Nicolai Lyholm                                                  #                                                   
# Date: 23/9/2024                                                                               #      
#                                                                                               #
#################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Write a code simulating a behavior of an M/M/1 system. Select the appropriate parameters
# for the arrival rate and the average service rate.

def mm1(service_rate, arrival_rate, num_customers):
    print(f"=====Running Simulation μ {service_rate}, λ {arrival_rate}, Customers {num_customers}=====")
    LastNotedDeparture = 0
    arrival_times = np.cumsum(np.random.exponential(1/arrival_rate, num_customers))
    service_times = np.random.exponential(1/service_rate, num_customers)
    departure_times = np.zeros(num_customers)

    delays = []
    queue_length = 0
    queue_lengths = []

    for i in tqdm(range(num_customers)):
        if i == 0:
            departure_times[i] = arrival_times[i] + service_times[i]
        else:
            departure_times[i] = max(arrival_times[i], departure_times[i-1]) + service_times[i]
        delays.append(max(0, departure_times[i] - arrival_times[i]))
        
        if arrival_times[i] < departure_times[i-1]:
            queue_length = 0
            last_last_departure = LastNotedDeparture
            for j in range(LastNotedDeparture, i):
                if arrival_times[i] < departure_times[j]:
                    queue_length += 1
                else:
                    LastNotedDeparture = j
            if arrival_times[i-1] < departure_times[last_last_departure] < arrival_times[i]:
                for q in range(queue_lengths[-1]-queue_length):
                    queue_lengths.append(queue_length-q)
            queue_lengths.append(queue_length)


        else:
            LastNotedDeparture = i-1
            queue_lengths.append(0)
# Theoretical results
    offered_load = arrival_rate/service_rate
    print(f"=============Theoretical results=============")
    print(f"Steady state probabilities:")
    theoretical_queue_length = []
    for i in range(max(queue_lengths)+1):
        theoretical_queue_length.append((1-offered_load)*(offered_load)**i)
        # print(f"Probability of {i} customers in the system: {(1-offered_load)*(offered_load)**i}")
    print(f"Customer throughput: {arrival_rate}")
    print(f"Average number of costumers: {offered_load/(1-offered_load)}" )
    print(f"Average delay: {1/(service_rate)+offered_load/(service_rate*(1-offered_load))}")
    print(f"Average waiting time in queue: {offered_load/(service_rate*(1-offered_load))}")
    print(f"Average queue length: {offered_load/(1-offered_load)}")

# Simulation results
    print(f"=============Simulation results=============")
    print(f"Customer throughput: {num_customers/departure_times[-1]}")
    print(f"Average delay: {np.mean(delays)}")
    print(f"Average waiting time in queue: {np.mean(delays-service_times)}")
    print(f"Average queue length: {np.mean(queue_lengths)}") ## Assumption we are in every que length for the same time
 

    plt.figure("RTC Curve")
    plt.step(arrival_times, np.arange(1, len(arrival_times)+1,1), label='Arrival curve')
    plt.step(departure_times, np.arange(1, len(departure_times)+1,1), label='Departure departure')
    plt.xlabel('Time')
    plt.ylabel('Number of customers')
    plt.title(f"RTC curves \n μ {service_rate}, λ {arrival_rate}, Customers {num_customers}")

    plt.legend()
    
    plt.figure(f'Delay curve')
    plt.hist(delays, bins=range(int(max(delays))+1),density=True)
    plt.title(f"Delay curve service \n μ {service_rate}, λ {arrival_rate}, Customers {num_customers}")
    plt.xlabel('Delay')
    plt.ylabel('Probability')

    
    plt.figure('Queue length curve')
    plt.title(f"Queue length curve \n μ: {service_rate}, λ: {arrival_rate}, Customers: {num_customers}")
    plt.step(np.arange(0,max(queue_lengths)+1)+1,theoretical_queue_length , label='Theoretical queue length')
    plt.hist(queue_lengths, bins=range(max(queue_lengths)+1),density=True,label='Simulation queue length')
    plt.xlabel('Number of customers in queue')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    mm1(service_rate=1.2, arrival_rate=1, num_customers=10**3)
    mm1(service_rate=1.2, arrival_rate=1, num_customers=10**5)
    mm1(service_rate=1.2, arrival_rate=1, num_customers=10**7)
    mm1(service_rate=2, arrival_rate=1, num_customers=10**7)
    
    