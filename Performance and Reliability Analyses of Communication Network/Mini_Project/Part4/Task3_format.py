from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.util import quietRun
import time
import subprocess
import os

class DumbellTopo(Topo):    
    def build(self,N,  **_opts):
        self.N = N
        bandwidth = 10
        delay= 5
        
        IP1 = '192.168.1.1/24'  # Ip Address for r0-eth1
        IP2 = '192.1.56.1/56'  # IP address for r1-eth1
        source = []
        dest = []
        routers = [self.addSwitch('R1'), self.addSwitch('R2')]
        self.addLink(routers[0],routers[1], bw = bandwidth, delay=delay, cls=TCLink, max_queue_size=21*delay, loss = 1)
        for i in range(self.N): 
            #Create Source and destination
            source.append(self.addHost("S{1}".format(i)))
            dest.append(self.addHost("D{1}".format(i)))
            # Create link 
            self.addLink(source[i],routers[0], bw = bandwidth, delay=delay, cls=TCLink, max_queue_size=21*delay, loss = 1)
            self.addLink(routers[1],dest[i], bw = bandwidth, delay=delay, cls=TCLink, max_queue_size=21*delay, loss = 1)

def cleanProbe():
    print("Removing existing TCP_Probe")
    procs = quietRun('pgrep -f /proc/net/tcpprobe').split()
    for proc in procs:
        output= quietRun('sudo kill -KILL {0}'.format(proc.rstrip()))

def TCP_prob(TCP_type,N):
    tcp_prob_dataPath = "{1}/{2}/TcpProbeData.txt".format(TCP_type,N)
    if(os.path.exists(tcp_prob_dataPath)):
        print("removing existing file")
        os.remove(tcp_prob_dataPath)
    cleanProbe()
    print("Starting TCP_PROBE........")
    output = quietRun('sudo rmmod tcp_probe')
    output = quietRun('sudo modprobe tcp_probe port=5001')
    print("Storing the TCP_Probe results")
    subprocess.Popen("sudo cat /proc/net/tcpprobe > {1}".format(tcp_prob_dataPath), shell=True)



global iperfDuration
iperfDuration = 30

def run(TCP_type, NuberOfSourceAndDest):
    "Test"
    topo = DumbellTopo(N = NuberOfSourceAndDest)
    datapath= "{1}/{2}/".format(TCP_type,N)
    net = Mininet(topo=topo)
    net.start()
    net.pingAll()
    source = [net.getNodeByName('S{1}'.format(i)) for i in range (topo.N)]
    dest = [net.getNodeByName('D{1}'.format(i)) for i in range (topo.N)]
    
    TCP_prob(TCP_type=TCP_type, N=NuberOfSourceAndDest)

    popens = dict()
    # net.iperf([source[1],dest[-1]],l4Type='TCP')
    for i in range(topo.N):
        print("Starting the server on the host S{1}".format(i))
        popens[source[i]]= source[i].popen(["iperf -s -p 5001 -i 1 -Z {1} > {datapath}S{2}.txt".format(TCP_type,i)], shell=True)
        
    time.sleep(1)
    
    for i in range(topo.N):
        popens[dest[i]]= dest[i].popen(["iperf -c {1} -p 5001 -t {2} > {3}D{4}.txt".format(source[i].IP(),iperfDuration,datapath,i)], shell=True)
    
    for i in range(topo.N):
           popens[dest[i]].wait()

    print("Done Iperf")
    #CLI(net)
    # os.system("sudo sh iperf3.sh")

    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    TCP_types=["reno","vegas","cubic"]
    for TCP_type in TCP_types:
        for N in range(1,10):
            if(not(os.path.exists("{1}/{2}".format(TCP_type,N)))):
                os.mkdir("{1}/{2}".format(TCP_type,N))
            run(TCP_type=TCP_type,NuberOfSourceAndDest=N)

