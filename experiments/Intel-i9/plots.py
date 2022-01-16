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




def plot_cake_skewed(fname = 'cake_skewed'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['K-first', 'M-first']
	#
	df1 = pandas.read_csv('results_KMN')
	df2 = pandas.read_csv('results_MKN')
	N = range(500,20001,500)
	KMN_M = [(2.0*i*576*576 / 1e9) / df1[(df1['sched'] == 'M') & (df1['s'] == i)]['time'].mean() for i in N]
	MKN_M = [(2.0*i*576*576 / 1e9) / df2[(df2['sched'] == 'M') & (df2['s'] == i)]['time'].mean() for i in N]
	KMN_K = [(2.0*i*576*576 / 1e9) / df1[(df1['sched'] == 'K') & (df1['s'] == i)]['time'].mean() for i in N]
	MKN_K = [(2.0*i*576*576 / 1e9) / df2[(df2['sched'] == 'K') & (df2['s'] == i)]['time'].mean() for i in N]
	plt.figure(figsize = (6,4))
	plt.plot(N, KMN_M, label = labels[0],  color = colors[0])
	plt.plot(N, MKN_M, label = labels[1],  color = colors[1])
	# plt.plot(list(NUM_CPUs), list(cake_mem_acc_arr), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	#
	plt.title('(a) Computation Throughput For Large M', fontsize = 16)
	plt.xlabel("M (K=N=576)", fontsize = 14)
	plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 14)
	plt.xticks(range(500,20001,4000))
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.savefig("%s_M.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(N, KMN_K, label = labels[0], color = colors[0])
	plt.plot(N, MKN_K, label = labels[1],  color = colors[1])
	# plt.plot(list(NUM_CPUs), list(cake_mem_acc_arr), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	#
	plt.title('(b) Computation Throughput For Large K', fontsize = 16)
	plt.xlabel("K (M=N=576)", fontsize = 14)
	plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 14)
	plt.xticks(range(500,20001,4000))
	plt.legend(loc = "lower right", prop={'size': 10})
	plt.savefig("%s_K.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')

	
plot_cake_skewed()
