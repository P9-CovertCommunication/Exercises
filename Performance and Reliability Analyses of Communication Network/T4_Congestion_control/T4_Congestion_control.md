# T4. Performance and Reliability Analysis of Communication Networks
## This is the property of the Authors, we gladly accept donations in the form of beer.
- Authors: Anders Bundgaard and Nicolai Lyholm  
- Date: 

## Task 1. What is congestion? Discuss ways to control congestion
*"Too many sources sending too much data too fast for the network to handle"* [Kurose, Ross]

Congestion differs from the notion of flow control in that we are now dealing with multiple users sharing a medium rather than just one transmitter overwhelming the receiver with a too fast transmission rate. Congestion can occur whenever devices have to share a medium with limited capacity. Typically congesion occurs due to a bottleneck (typically a router), which has a limited buffer size and can only process packets at a given capacity. Consequences of congestion include:
* Long delays (queueing in router waiting to be serviced)
* Packet loss (buffer overflow at routers)

Congestion occurs if the incoming packet rate exceeds the outgoing rate, that is $\lambda_{in}>\lambda_{out}$ or if considering retransmissions $\lambda_{in}'>\lambda_{out}$. Several strategies and parameters can be tweaked to control congestion without relying on TCP specific mechanisms like TCP Reno or Tahoe.
* **Increase capacity** - Naturally a way to avoid congestion is to just invest in more capacity or bandwidth. Meaning the network can handle more incoming packets quicker. Of course this is both expensive and not always a feasible solution.
* **Traffic shaping** - Intelligently scheduling transmissions such that non-urgent data is transmitted during off-peak times can help alleviate congestion, typically a network will experience more traffic during cetrain times of day, if the load can be distributed more evenly this can alleviate the congestion.
* **Perfect knowledge** - If users have perfect knowledge of the network, that means current buffer conditions, capacity and knowledge of all users on the network, the transmissions can simply be scheduled in a way which avoids transmitting while the buffer is full thereby not needing any retransmissions and reducing congestion.
* **Optimal routing** - In a network consisting of multiple routers functioning in a packet switched manner, congesion can often occur if many users are routed through the same router causing a bottleneck. Therefore routing in such a way, which utilizes the capacity of all routers will minimize congestion.

![alt text](Congestion_system.png)

## Task 2. What is the bandwidth-delay product? Explain its relation with congestion.
The bandwidth-delay product (BDP) is a key concept in networking, helping us understand the relationship between network capacity and inherent delay. The capacity or bandwidth of a network is typically denoted in bits per second [bps] and the delay is measured in the form of round-trip-time (RTT). The BDP is given as:
$$ \textrm{BDP} = BW\cdot RTT$$

The BDP represents the maximum amount of bits that can be in transit in the network at any given time. It represents the ideal transmission rate, which will fully utilize the network without causing congestion. If the transmitter exceeds the BDP buffers will start to fill up causing delays and potentially packet loss due to congestion.

## Task 3. Explain the AIMD approach of TCP. Discuss Slow Start, Collision Avoidance, and Fast Recovery phases.

## Task 4. Discuss TCP Reno and TC Tahoe.

## Task 5. What are the signals to indicate congestion?

## Task 6. What is the fairness problem in TCP congestion control?

## Task 7. Provide an overview of BBR, AQM, and ECN options.