# Task 1. 
## Outline differences between circuit-switching and packet-switching.
* **Connection establishment**
    - Circuitswitching: A logical connection is required between transmitter and receiver, this communication path is established before data transmission and remains active for the duration of communication. (Other devices cannot use the same communication path)
    - Packet-switching: There is no dedicated communication path, large chunks of data is packetized and each packet is routed through the network independently. The data packets are then organized at the receiver and the original data is reconstructed.

* **Data transmission**
    - Circuit-switching: Data is transmitted continously along the pre-determined path ensuring sequential delivery.
    - Packet-switching: Data is packetized and the arrival of packets happen independently with no strict order. The original data is then reassembled at the receiver.


* **Reliablity**
    - The reliablity of circuit-switching is higher as the resource is reservered to you. An example is a phone call is better quality than a messenger call, since the communication path is reserved for the duration of the phone call.   

# Task 2. 
## What is a counting process? Which distribution is used to model the counting process? 
- A Counting process is a random process that only can increass


# Task 3.
## What are basic HTTP requests?


# Task 4.
## Analyze the HTTP trace. You can use “Wireshark”.
**• Get the HTTP trace: https://kevincurran.org/com320/labs/wireshark/trace-http.pcap** 
**• Filter the HTTP traffic.**
**• Find an HTTP request (basically, get the first GET packet and expand its block).**
## Write the source and destination IP.
**• Show the headers and comment on what they mean. (e.g., “Host”: this is a mandatory header that identifies the name and port of the server)**
**• Show response to the first GET in the trace; expand the block. Report “Status Code” and “Status Code Description”.**
**• Can you verify if the server needs to send fresh content for the second GET request? (Hint: check the third and fourth HTTP trace; check the “if-modified-since” header.)**

# Task 5.
## Download the exercise program from https://luca31.github.io/HTTP-requests-exercises/ and post the code and results obtained in the report. Print the POST requests and the relative responses in full, including the URL, body, and headers. What request code numbers do the actual packets use?
If you get an “unhandled error event” when you start the server, open index.js and
comment the three lines after the comment “copy to clipboard,” i.e., lines 101-103 in the
file. The server should work then.