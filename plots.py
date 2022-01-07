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
	# df1 = pandas.read_csv('inner_prod_power.csv')
	df1 = pandas.read_csv('arm_vs_mema_fp32.csv')
	# N = [1024,2048,4096,8192]
	N1 = range(5,111,5)
	N2 = range(8,111,8)
	# energy1 = [df1[df1['N'] == i]['kernel'].mean()  for i in N]
	tput_inner_1x16x1 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_1x16x1') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	tput_inner_2x8x2 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_2x8x2') \
		& (df1['M'] == i)]['time'].values[0]) for i in N2]
	tput_mema = [float(2*(i**3)) / float(df1[(df1['algo'] == 'mema') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	fig = plt.figure(figsize = (6,4))
	plt.title('FP32 Computation Throughput of CMSIS-DSP vs MEMA', fontsize = 14)
	plt.plot(N1, tput_mema, 'b', label = 'mema_outer_5x5', marker = markers[4], color = colors[4])
	plt.plot(N2, tput_inner_2x8x2, 'b', label = 'arm_inner_2x8x2', marker = markers[3], color = colors[3])
	plt.plot(N1, tput_inner_1x16x1, 'b', label = 'arm_inner_1x16x1', marker = markers[1], color = colors[1])
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.xlabel('N', fontsize = 16)
	plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 14)
	plt.xticks(range(0,111,10),fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')




plot_arm_vs_mema_fp32_tput()





def plot_arm_vs_mema_q15_tput(fname = 'arm_vs_mema_q15_tput'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	# df1 = pandas.read_csv('inner_prod_power.csv')
	df1 = pandas.read_csv('arm_vs_mema.csv')
	# N = [1024,2048,4096,8192]
	N1 = range(5,111,5)
	N2 = range(8,111,8)
	# energy1 = [df1[df1['N'] == i]['kernel'].mean()  for i in N]
	tput_inner_1x16x1 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_1x16x1') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	tput_inner_2x8x2 = [float(2*(i**3)) / float(df1[(df1['algo'] == 'inner_2x8x2') \
		& (df1['M'] == i)]['time'].values[0]) for i in N2]
	tput_mema = [float(2*(i**3)) / float(df1[(df1['algo'] == 'mema') \
		& (df1['M'] == i)]['time'].values[0]) for i in N1]
	fig = plt.figure(figsize = (6,4))
	plt.title('FP32 Computation Throughput of CMSIS-DSP vs MEMA', fontsize = 14)
	plt.plot(N1, tput_mema, 'b', label = 'mema_outer_5x5', marker = markers[4], color = colors[4])
	plt.plot(N2, tput_inner_2x8x2, 'b', label = 'arm_inner_2x8x2', marker = markers[3], color = colors[3])
	plt.plot(N1, tput_inner_1x16x1, 'b', label = 'arm_inner_1x16x1', marker = markers[1], color = colors[1])
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.xlabel('N', fontsize = 16)
	plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 14)
	plt.xticks(range(0,111,10),fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')




plot_arm_vs_mema_q15_tput()



