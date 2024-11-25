import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sport', help="Enable the source port filter (Default is dest port)", action='store_true', dest="sport", default=False)
parser.add_argument('-p', '--port', dest="port", default='5001')
parser.add_argument('-f', dest="files", nargs='+', default="Deafultout.png")
parser.add_argument('-o', '--out', dest="out", default=None)
parser.add_argument('-t', '--tcpType', dest="tcpType", default=None)
parser.add_argument('-H', '--histogram', dest="histogram",
                    help="Plot histogram of sum(cwnd_i)",
                    action="store_true",
                    default=False)

args = parser.parse_args()

def first(lst):
    return [e[0] for e in lst]

def second(lst):
    return [e[1] for e in lst]

def parse_file(f):
    times = defaultdict(list)
    cwnd = defaultdict(list)
    srtt = []
    with open(f) as file:
        for l in file:
            fields = l.strip().split(' ')
            if len(fields) < 10:
                break
            if not args.sport:
                if fields[2].split(':')[1] != args.port:
                    continue
            else:
                if fields[1].split(':')[1] != args.port:
                    continue
            sport = int(fields[1].split(':')[1])
            times[sport].append(float(fields[0]))
            c = int(fields[6])
            cwnd[sport].append(c * 1480 / 1024.0)
            srtt.append(int(fields[-1]))
    return times, cwnd

added = defaultdict(int)
events = []

def plot_cwnds(ax):
    global events
    for f in args.files:
        times, cwnds = parse_file(f)
        for port in sorted(cwnds.keys()):
            t = times[port]
            cwnd = cwnds[port]
            events += zip(t, [port]*len(t), cwnd)
            ax.plot(t, cwnd, label=f"cwnd_{port}")
    events.sort()

total_cwnd = 0
cwnd_time = []
totalcwnds = []

fig = plt.figure()
plots = 1
if args.histogram:
    plots = 2

axPlot = fig.add_subplot(1, plots, 1)
plot_cwnds(axPlot)

for (t, p, c) in events:
    if added[p]:
        total_cwnd -= added[p]
    total_cwnd += c
    cwnd_time.append((t, total_cwnd))
    added[p] = c
    totalcwnds.append(total_cwnd)
axPlot.plot(first(cwnd_time), second(cwnd_time), lw=2, label="$\sum_i W_i$")
axPlot.grid(True)
axPlot.set_xlabel("seconds")
axPlot.set_ylabel("cwnd KB")
if args.tcpType:
    axPlot.set_title(f"TCP-{args.tcpType} congestion window (cwnd) timeseries")
axPlot.legend()

if args.histogram:
    axHist = fig.add_subplot(1, 2, 2)
    n, bins, patches = axHist.hist(totalcwnds, 50, density=True, facecolor='green', alpha=0.75)
    axHist.set_xlabel("bins (KB)")
    axHist.set_ylabel("Fraction")
    axHist.set_title("Histogram of sum(cwnd_i)")

if args.out:
    print(f'saving to {args.out}')
    plt.savefig(args.out)
else:
    plt.show()