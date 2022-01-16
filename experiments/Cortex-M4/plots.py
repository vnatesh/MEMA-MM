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





def plot_arm_vs_mema_fp32_tput(fname = 'arm_vs_mema_fp32_tput'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	df1 = pandas.read_csv('arm_vs_mema_fp32.csv')
	N1 = range(5,111,5)
	N2 = range(8,111,8)
	tput_inner_1x16x1 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_1x16x1') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	tput_inner_2x8x2 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_2x8x2') \
		& (df1['M'] == i)]['time'].values[0]) for i in N2]
	tput_mema = [float(2*(i**3)) / float(df1[(df1['algo'] == 'mema') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	fig = plt.figure(figsize = (6,4))
	plt.title('(a) FP32 Throughput of MEMA vs ARM MCU', fontsize = 14)
	plt.plot(N1, tput_mema, 'b', label = 'mema_outer_5x5', marker = markers[4], color = colors[4])
	plt.plot(N2, tput_inner_2x8x2, 'b', label = 'arm_inner_2x8x2', marker = markers[1], color = colors[1])
	plt.plot(N1, tput_inner_1x16x1, 'b', label = 'arm_inner_1x16x1', marker = markers[3], color = colors[3])
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.xlabel('N', fontsize = 16)
	plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 14)
	plt.xticks(range(0,111,20),fontsize = 14)
	plt.yticks(fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



def plot_arm_vs_mema_q15_tput(fname = 'arm_vs_mema_q15_tput'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	df1 = pandas.read_csv('arm_vs_mema_q15.csv')
	N1 = range(8,111,8)
	tput_inner_2x4x2 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_2x4x2') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	tput_inner_2x2x2 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_2x2x2') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	tput_mema = [float(2*(i**3)) / float(df1[(df1['algo'] == 'outer_q15_4x2') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	fig = plt.figure(figsize = (6,4))
	plt.title('(c) Q15 Throughput of MEMA vs ARM MCU', fontsize = 14)
	plt.plot(N1, tput_mema, 'b', label = 'mema_outer_4x2x2', marker = markers[4], color = colors[4])
	# plt.plot(N1, tput_inner_2x2x2, 'b', label = 'arm_inner_2x2x2', marker = markers[3], color = colors[3])
	plt.plot(N1, tput_inner_2x4x2, 'b', label = 'arm_inner_2x4x2', marker = markers[1], color = colors[1])
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.xlabel('N', fontsize = 16)
	plt.ylabel('Throughput (MIOPs/sec)', fontsize = 14)
	plt.xticks(range(0,111,20),fontsize = 14)
	plt.yticks(fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



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



def plot_arm_vs_mema_energy(fname = 'arm_vs_mema_energy'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	df1 = pandas.read_csv('power_outer_fp32.csv')
	N1 = range(10,111,10)
	N2 = range(8,111,8)
	#
	out_fp32 = compute_energy_peaks('power_outer_fp32.csv', 20, 18)
	in_fp32 = compute_energy_peaks('power_inner_fp32.csv', 20, 18)
	out_q15 = compute_energy_peaks('power_outer_q15.csv', 20, 18)
	in_q15 = compute_energy_peaks('power_inner_q15.csv', 20, 18)
	#
	for i in range(len(N1)):
		if N1[i] <= 50:
			out_fp32[i] /= 1000
			in_fp32[i] /= 1000
		else:
			out_fp32[i] /= 500 
			in_fp32[i] /= 500 
	#
	for i in range(len(N2)):
		if N2[i] <= 56:
			out_q15[i] /= 1000
			in_q15[i] /= 1000
		else:
			out_q15[i] /= 500 
			in_q15[i] /= 500 
	#
	fp32_ops = [2.0*(i**3) / 1e6 for i in N1]
	q15_ops = [2.0*(i**3) / 1e6 for i in N2]
	#
	fig = plt.figure(figsize = (6,4))
	plt.title('(b) FP32 Energy Usage in MEMA vs ARM MCU', fontsize = 14)
	plt.plot(fp32_ops, in_fp32, 'b', label = 'arm_inner_2x8x2', marker = markers[1], color = colors[1])
	plt.plot(fp32_ops, out_fp32, 'b', label = 'mema_outer_5x5', marker = markers[4], color = colors[4])
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.xlabel('Number of Operations (MFLOPs)', fontsize = 16)
	plt.ylabel('Energy Usage (mJ)', fontsize = 14)
	plt.xticks(fontsize = 14)
	plt.yticks(fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s_fp32.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	fig = plt.figure(figsize = (6,4))
	plt.title('(d) Q15 Energy Usage in MEMA vs ARM MCU', fontsize = 14)
	plt.plot(q15_ops, in_q15, 'b', label = 'arm_inner_2x4x2', marker = markers[1], color = colors[1])
	plt.plot(q15_ops, out_q15, 'b', label = 'mema_outer_4x2x2', marker = markers[4], color = colors[4])
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.xlabel('Number of Operations (MIOPs)', fontsize = 16)
	plt.ylabel('Energy Usage (mJ)', fontsize = 14)
	plt.xticks(fontsize = 14)
	plt.yticks(fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s_q15.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



plot_arm_vs_mema_q15_tput()
plot_arm_vs_mema_fp32_tput()
plot_arm_vs_mema_energy(fname = 'arm_vs_mema_energy')






