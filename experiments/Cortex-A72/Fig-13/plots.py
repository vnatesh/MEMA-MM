import numpy as np
from scipy import optimize
import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as mtick
import numpy as np
import os
import re
import math




# integrate power over time (200 us interval) to get energy during MM 
def compute_energy_peaks(fname, high, low):
	df1 = pandas.read_csv(fname)
	c = list(df1['USB Avg Current (mA)'])
	p = list(df1['USB Avg Power (W)'])
	i = 0;
	energy = []
	while i < len(p):
		if c[i] >= high:
			x = 0
			j = 0
			while c[i+j] >= low:
				# energy (J) = sum(power (W) X time (sec)) (0.0002 sec per sample)
				x += p[i+j]*0.0002
				j += 1
			#
			energy.append(x * 1000.0)
			i += j
		else:
			i += 1
	#
	return energy



def plot_cake_vs_arm_tput(fname = 'cake_vs_arm_tput'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['MEMA', 'ARMPL','ARMCL']
	df1 = pandas.read_csv('cake_vs_arm_results')
	N = range(500,5001,500)
	ops = [2.0*i*i*i / 1e9 for i in N]
	armpl = [float(2*(i**3) / 1e9 ) / float(df1[(df1['algo'] == 'armpl') \
		& (df1['M'] == i)]['time'].values[0]) for i in N]
	armcl = [float(2*(i**3) / 1e9 ) / float(df1[(df1['algo'] == 'armcl') \
		& (df1['M'] == i)]['time'].values[0]) for i in N]
	cake = [float(2*(i**3) / 1e9 ) / float(df1[(df1['algo'] == 'cake') \
		& (df1['M'] == i)]['time'].values[0]) for i in N]
	fig = plt.figure(figsize = (6,4))
	plt.title('(a) Computation Throughput of MEMA vs ARM CPU', fontsize = 14)
	plt.plot(list(ops), armcl, label = labels[2],  marker = markers[2], color = colors[3])
	plt.plot(list(ops), armpl, label = labels[1],  marker = markers[1], color = colors[5])
	plt.plot(list(ops), cake, label = labels[0],  marker = markers[0], color = colors[4])
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.xlabel('Number of Operations (GFLOPs)', fontsize = 18)
	plt.ylabel('Throughput (GFLOPs/sec)', fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_cake_vs_arm_tput()



def plot_cake_vs_arm_io(fname = 'cake_vs_arm_io'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['MEMA', 'ARMPL','ARMCL']
	io_cake = []; io_armpl = []; io_armcl = []; 
	N = range(500,5001,500)
	ops = [2.0*i*i*i / 1e9 for i in N]
	#
	for i in N:
		# multiply by 64 bytes since external memory request non-cacheable 
		# and L2-data cache refills/writeback PMUs
		# in ARM are expressed in terms of number of cache lines
		a = open('reports_arm_cnn/report_armpl_%d' % i,'r').read().split('\n')
		io = ((int(re.search(r'\d+', a[5]).group())*64.0) ) / 1e9
		io += ((int(re.search(r'\d+', a[6]).group())*64.0)) / 1e9
		io_armpl.append(io)
		#
		a = open('reports_arm_cnn/report_armcl_%d' % i,'r').read().split('\n')
		io = ((int(re.search(r'\d+', a[5]).group())*64.0) ) / 1e9
		io += ((int(re.search(r'\d+', a[6]).group())*64.0)) / 1e9
		io_armcl.append(io)
		#
		a = open('reports_arm_cnn/report_cake_%d' % i,'r').read().split('\n')
		io = ((int(re.search(r'\d+', a[5]).group())*64.0) ) / 1e9
		io += ((int(re.search(r'\d+', a[6]).group())*64.0)) / 1e9
		io_cake.append(io)
	#
	# plt.subplot(1, 2, 1)
	plt.figure(figsize = (6,4))
	plt.plot(list(ops), list(io_armcl), label = labels[2],  marker = markers[2], color = colors[3])
	plt.plot(list(ops), list(io_armpl), label = labels[1],  marker = markers[1], color = colors[5])
	plt.plot(list(ops), list(io_cake), label = labels[0],  marker = markers[0], color = colors[4])
	#
	plt.title('(b) DRAM IO of MEMA vs ARM CPU')
	plt.xlabel("Number of Operations (GFLOPs)", fontsize = 18)
	plt.ylabel("DRAM IO (GB)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 10})
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



plot_cake_vs_arm_io()




len(compute_energy_peaks('power_armpl.csv', 800, 550))
len(compute_energy_peaks('power_armcl.csv', 800, 550))
len(compute_energy_peaks('power_cake.csv', 800, 550))

compute_energy_peaks('power_armpl.csv', 800, 550)
compute_energy_peaks('power_armcl.csv', 800, 550)
compute_energy_peaks('power_cake.csv', 800, 550)

def plot_cake_vs_arm_energy(fname = 'cake_vs_arm_energy'):
	armpl = [83.81293999999994, 677.6829599999996, 2519.1247599999956, 5692.071219999987, 10828.513979999869, 19358.428860000018, 30004.140100000397, 47787.191260001084, 65019.56523999948, 88182.4322]
	armcl = [74.97116, 691.3995000000001, 2447.699639999996, 5334.108280000018, 9725.845560000005, 17350.552020000097, 26106.486600000277, 40835.75489999931, 53762.85884000163, 75192.62882000238]
	cake = [87.53432000000001, 616.9226800000004, 2095.7021400000053, 4585.453020000009, 8740.65449999992, 15267.011400000032, 23376.397, 34520.006019999004, 49919.2141599992, 68352.945]
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['MEMA', 'ARMPL','ARMCL']
	N = range(500,5001,500)
	ops = [2.0*i*i*i / 1e9 for i in N]
	#
	plt.figure(figsize = (6,4))
	plt.plot(list(ops), [i/1000 for i in armcl], label = labels[2],  marker = markers[2], color = colors[3])
	plt.plot(list(ops), [i/1000 for i in armpl], label = labels[1],  marker = markers[1], color = colors[5])
	plt.plot(list(ops), [i/1000 for i in cake], label = labels[0],  marker = markers[0], color = colors[4])
	#
	plt.title('(c) Energy Usage of MEMA vs ARM CPU')
	plt.xlabel("Number of Operations (GFLOPs)", fontsize = 18)
	plt.ylabel("Energy Usage (J)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 10})
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_cake_vs_arm_energy()
