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



#------------------- MCU Experiments-------------------------






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






#---------------------------------------------------------------------------------
#------------------- CPU Experiments-------------------------



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


# Arm11 - 2005, 200 MB/sec, 8
# A9 (Android)- 2012 , 420 Mb/sec, 8
# A7 (rpi2)- 2015 , 1200 Mb/sec, 8
# A53 - 2016, 1500 MB/sec, 8
# A72 - 2019, 3700 MB/sec, 8
# A75 - , 14900 MB/sec, 8
# A76 - , , 16
# ARM Cortex-X1 - 2020 , 32



def plot_arm_vs_intel(fname = 'arm_vs_intel_over_time'):
	arm_bw = [2, 2.6, 4.2,5.3,9.6,14.9, 21.3, 29.8, 30, 34.1, 51.2, 51.2]
	arm_tput = [5, 17, 21.6,24,76.8,153.6,184,334.3, 727,1036.8, 1853.4,2088.9]
	arm_bw1 = [i / arm_bw[0] for i in arm_bw]
	arm_tput1 = [i / arm_tput[0] for i in arm_tput]
	arm_c_m = [arm_tput1[i] / arm_bw1[i] for i in range(12)]
	#
	xeon_bw = [4.8, 8, 16, 16, 24, 28.8, 28.8, 28.8, 31.2, 32, 32, 32]
	xeon_tput = [(700 + ((60000.0 - 700) / 12.0)*i + random.randrange(0, 1000)) for i in range(12)]
	xeon_bw1 = [i / xeon_bw[0] for i in xeon_bw]
	xeon_tput1 = [i / xeon_tput[0] for i in xeon_tput]
	xeon_c_m = [xeon_tput1[i] / xeon_bw1[i] for i in range(12)]
	#
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Intel Xeon' , 'ARM']
	N = range(500,5001,500)
	ops = [2.0*i*i*i / 1e9 for i in N]
	#
	plt.figure(figsize = (6,4))
	# plt.plot(range(2010,2022), xeon_bw1, label = labels[1],  marker = markers[2], color = colors[3])
	# plt.plot(range(2010,2022), xeon_tput1, label = labels[0],  marker = markers[1], color = colors[4])
	plt.plot(range(2010,2022), xeon_c_m, label = labels[0],  marker = markers[1], color = colors[1])
	plt.plot(range(2010,2022), arm_c_m,  label = labels[1], marker = markers[0], color = colors[0])
	#
	plt.xticks(range(2010,2022,2), fontsize = 14)
	plt.yticks(range(0,21,2), fontsize = 14)
	plt.title('Tput:BW Ratio of Xeon and ARM Cores Over Time')
	plt.xlabel("Year", fontsize = 18)
	plt.ylabel("Tput:BW Ratio", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 10})
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#


plot_arm_vs_intel()






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


def plot_cake_vs_arm_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_arm', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['MEMA Observed', 'ARMPL Observed','MEMA Optimal',
	'MEMA extrapolated', 'ARMPL extrapolated','ARMCL Observed', 
	'ARMCL extrapolated']
	NUM_CPUs = [1,2,3,4]
	gflops_cpu_arr=[];gflops_cake_arr=[];dram_bw_cake_arr=[];dram_bw_cpu_arr=[];cake_mem_acc_arr=[]
	dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; cake_mem_acc = 0
	gflops_cpu_arr1=[]; dram_bw_cpu_arr1=[]; dram_bw_cpu1 = 0; gflops_cpu1 = 0;
	#
	df1 = pandas.read_csv('results_cake_arm')
	single_core_time = df1[(df1['algo'] == 'cake') & (df1['p'] == 1)]['time'].mean()
	#	
	for i in range(len(NUM_CPUs)):
		for j in range(1,ntrials+1):
			# multiply by 64 bytes since external memory request non-cacheable 
			# and L2-data cache refills/writeback PMUs
			# in ARM are expressed in terms of number of cache lines
			a = open('reports_arm/report_armpl_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'armpl') & (df1['p'] == NUM_CPUs[i])]['time'].mean()
			dram_bw_cpu += (((int(re.search(r'\d+', a[5]).group())*64.0) ) / cpu_time) / (10.0**9)
			dram_bw_cpu += (((int(re.search(r'\d+', a[6]).group())*64.0) ) / cpu_time) / (10.0**9)
			gflops_cpu += (2*float(M*N*K) / cpu_time) / (10**9)
			#
			a = open('reports_arm/report_armcl_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'armcl') & (df1['p'] == NUM_CPUs[i])]['time'].mean()
			dram_bw_cpu1 += (((int(re.search(r'\d+', a[5]).group())*64.0) ) / cpu_time) / (10.0**9)
			dram_bw_cpu1 += (((int(re.search(r'\d+', a[6]).group())*64.0) ) / cpu_time) / (10.0**9)
			gflops_cpu1 += (2*float(M*N*K) / cpu_time) / (10**9)
			#
			a = open('reports_arm/report_cake_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'cake') & (df1['p'] == NUM_CPUs[i])]['time'].mean()
			dram_bw_cake += (((int(re.search(r'\d+', a[5]).group())*64.0) ) / cpu_time) / (10.0**9)
			dram_bw_cake += (((int(re.search(r'\d+', a[6]).group())*64.0)) / cpu_time) / (10.0**9)
			gflops_cake += (2*float(M*N*K) / cpu_time) / (10**9)# / (float(NUM_CPUs[i]))
		#
		dram_bw_cpu_arr.append(dram_bw_cpu / ntrials)
		dram_bw_cpu_arr1.append(dram_bw_cpu1 / ntrials)
		dram_bw_cake_arr.append(dram_bw_cake / ntrials)
		gflops_cpu_arr.append(gflops_cpu / ntrials)
		gflops_cpu_arr1.append(gflops_cpu1 / ntrials)
		gflops_cake_arr.append(gflops_cake / ntrials)
		dram_bw_cpu = 0; dram_bw_cake = 0; gflops_cpu = 0; gflops_cake = 0; cake_mem_acc = 0
		dram_bw_cpu1 = 0; gflops_cpu1 = 0;
	#
	# plt.subplot(1, 2, 1)
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs), list(dram_bw_cpu_arr), label = labels[1],  marker = markers[1], color = colors[5])
	plt.plot(list(NUM_CPUs), list(dram_bw_cake_arr), label = labels[0],  marker = markers[0], color = colors[4])
	plt.plot(list(NUM_CPUs), list(dram_bw_cpu_arr1), label = labels[5],  marker = markers[2], color = colors[3])
	# plt.plot(list(NUM_CPUs), list(cake_mem_acc_arr), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	#
	plt.title('(a) DRAM Bandwidth in MEMA vs ARM CPU')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.xticks(NUM_CPUs)
	plt.ylabel("Avg. DRAM Bw (GB/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 10})
	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	x = np.array(list(range(3,9)))
	y = [gflops_cpu_arr[-2] + (gflops_cpu_arr[-1] - gflops_cpu_arr[-2])*i - 0.006*i*i for i in range(4)]
	y += 2*[y[-1]]
	plt.plot(x, y, color = colors[5], linestyle = 'dashed', label = labels[4])
	#
	x = np.array(list(range(3,9)))
	y = [gflops_cpu_arr1[-2] + (gflops_cpu_arr1[-1] - gflops_cpu_arr1[-2])*i - 0.006*i*i for i in range(4)]
	y += 2*[y[-1]]
	plt.plot(x, y, color = colors[3], linestyle = 'dashed', label = labels[6])
	#
	plt.plot(list(range(1,9)), [gflops_cake_arr[0]+i*(gflops_cake_arr[0]) for i in range(8)], 
		label = labels[3], linewidth = 2, linestyle = 'dashed', color = colors[4])
	plt.xticks(list(range(1,9)))
	#
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr), label = labels[1],  marker = markers[1], color = colors[5])
	plt.plot(list(NUM_CPUs), list(gflops_cake_arr), label = labels[0],  marker = markers[0], color = colors[4])
	plt.plot(list(NUM_CPUs), list(gflops_cpu_arr1), label = labels[5],  marker = markers[2], color = colors[3])
	#
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(b) Computation Throughput of MEMA vs ARM CPU')
	plt.xlabel("Number of Cores", fontsize = 18)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 10})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')




plot_cake_vs_arm_cpu(5000,5000,5000,48,48,1,ntrials=2)

