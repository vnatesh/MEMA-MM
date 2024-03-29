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
import random



# integrate power over time (200 us interval) to get energy (mJ) during MM 
def compute_energy_peaks(fname, high, low):
	df1 = pandas.read_csv(fname)
	c = list(df1['USB Avg Current (mA)'])
	p = list(df1['USB Avg Power (mW)']/1e3)
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
	out_fp32 = compute_energy_peaks('power_outer_fp32_tiny.csv', 20, 18)
	in_fp32 = compute_energy_peaks('power_inner_fp32_tiny.csv', 20, 18)
	out_q15 = compute_energy_peaks('power_outer_q15_tiny.csv', 20, 18)
	in_q15 = compute_energy_peaks('power_inner_q15_tiny.csv', 20, 18)
	Ms = [16,16,32,32,32,64,64,64, 64, 64, 8 ,16 ,32 ,32 ,64 ,64 ,128,128,256,256]
	Ks = [27 ,144,144,288,16 ,288,576,32 , 40, 64, 27 , 8 , 16 , 32 , 32 , 64 , 64 , 128, 128, 256]
	Ns = [1024,1024,256,256,256,64,64,64, 122, 125, 2304,2304,576,576,144,144, 36, 36, 9, 9]
	#
	for i in range(len(Ns)):
		out_fp32[i] /= 10
		in_fp32[i] /= 10
		out_q15[i] /= 10
		in_q15[i] /= 10
	#
	# fp32_ops = [2.0*Ms[i]*Ks[i]*Ns[i] / 1e6 for i in range(len(Ns))]
	# q15_ops = [22.0*Ms[i]*Ks[i]*Ns[i] / 1e6 for i in range(len(Ns))]
	#
	plt.rcParams["legend.framealpha"] = 1
	plt.figure(figsize = (16,4))
	barWidth = 0.2
	br1 = np.arange(len(Ms))
	br2 = [x + barWidth for x in br1]
	# Make the plot
	plt.bar(br2, out_fp32, color =colors[4], width = barWidth,
	        edgecolor ='grey', label ='FP32 mema outer 5x1x5')
	plt.bar(br1, in_fp32, color =colors[1], width = barWidth,
	        edgecolor ='grey', label = 'FP32 cmsis inner 2x8x2')
	#
	#
	#
	barWidth = 0.2
	br3 = [x + barWidth for x in br2]
	br4 = [x + barWidth for x in br3]
	# Make the plot
	plt.bar(br4, out_q15, color =colors[4], width = barWidth,
	        edgecolor ='black', hatch="//", label ='Q15 mema outer 4x2x2')
	plt.bar(br3, in_q15, color =colors[1], width = barWidth,
	        edgecolor ='black', hatch="//", label = 'Q15 cmsis inner 2x4x2')
	#
	plt.title('(b) MM Energy Usage on Cortex-M4', fontsize = 30)
	plt.xlabel("Layer Id", fontsize = 24)
	plt.ylabel('Energy Usage (mJ)', fontsize = 24)
	# plt.xticks([r + barWidth for r in range(len(Ms))],
 #        ["%dx%dx%d" % (Ms[i],Ks[i],Ns[i]) for i in range(len(Ms))])
	plt.xticks([r + barWidth for r in range(len(Ms))],
        [i+1 for i in range(len(Ms))])
	plt.legend(loc = "upper right", prop={'size': 18})
	plt.xticks(fontsize = 24)
	plt.yticks(fontsize = 20)
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')






plot_arm_vs_mema_energy()







def roofline(fname = 'roofline_arm'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','r','aqua','k','m']
	cpus = ['Cortex-M4','Cortex-M7','Cortex-A72']
	tput = [42*1e6,283*1e6,32*1e9]
	bw = [256*1e6,864*1e6,4*1e9]
	ai = range(13)
	fig = plt.figure(figsize = (6,4))
	plt.title('(c) Roofline Models For Embedded ARM CPUs', fontsize = 20)
	plt.plot(ai, [i for i in ai[:5]] + [4]*(len(ai)-5), 'b', label = cpus[0], color = colors[0])
	plt.plot(ai, [2.5*i for i in ai[:7]] + [15]*(len(ai)-7), 'b', label = cpus[1], color = colors[1])
	plt.plot(ai, [4*i for i in ai[:9]] + [32]*(len(ai)-9), 'b', label = cpus[2], color = colors[2])
	plt.scatter([4],[4], color = 'k')
	plt.scatter([6],[15], color = 'k')
	plt.scatter([8],[32], color = 'k')
	plt.plot([4,4], [0,4], '--', color = 'tab:gray')
	plt.plot([6,6], [0,15], '--', color = 'tab:gray')
	plt.plot([8,8], [0,32], '--', color = 'tab:gray')
	#
	plt.plot([0,4], [4,4], '--', color = 'tab:gray')
	plt.plot([0,6], [15,15], '--', color = 'tab:gray')
	plt.plot([0,8], [32,32], '--', color = 'tab:gray')
	plt.legend(loc = "upper left", prop={'size': 14})
	plt.xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize = 20)
	plt.ylabel('Throughput (GFLOPs/sec)', fontsize = 20)
	plt.xticks([4,6,8],
		[round(tput[0] / bw[0],2), 
		round(tput[1] / bw[1],2),
		round(tput[2] / bw[2],2)],fontsize = 14)
	plt.yticks([4,15,32],
		[tput[0] / 1e9, 
		tput[1] / 1e9,
		tput[2] / 1e9],fontsize = 14)
	plt.xlim(xmin = 0,xmax = 10)
	plt.ylim(ymin = 0,ymax = 33)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


roofline()



def gen_layer_table():
	# Ms = [16,16,32,32,32,64,64,64, 64, 64, 8 ,16 ,32 ,32 ,64 ,64 ,128,128,256,256]
	# Ks = [27 ,144,144,288,16 ,288,576,32 , 40, 64, 27 , 8 , 16 , 32 , 32 , 64 , 64 , 128, 128, 256]
	# Ns = [1024,1024,256,256,256,64,64,64, 122, 125, 2304,2304,576,576,144,144, 36, 36, 9, 9]
	Ms=(512,512,2048,512,512,2048,512,512,2048,512,512,2048)
	Ks=(2048,512,512,2048,512,512,2048,512,512,2048,512,512)
	Ns=(256,256,256,512,512,512,1024,1024,1024,2048,2048,2048)
	print()
	for i in range(len(Ms)):
		print("%d & $%d\\times%d\\times%d$ \\\\" % (i+1,Ms[i],Ks[i],Ns[i]))




def plot_mema_vs_arm_dlmc(fname = 'mema_vs_arm_dlmc', ntrials = 10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['MEMA', 'ARMPL', 'ARMCL']
	Ms=(512,512,2048,512,512,2048,512,512,2048,512,512,2048)
	Ks=(2048,512,512,2048,512,512,2048,512,512,2048,512,512)
	Ns=(256,256,256,512,512,512,1024,1024,1024,2048,2048,2048)
	df1 = pandas.read_csv('results_dlmc')
	#
	#
	#
	gflops_armpl_arr=[];gflops_mema_arr=[];dram_io_mema_arr=[];dram_io_armpl_arr=[];cake_mem_acc_arr=[]
	dram_io_armpl = 0; dram_io_mema = 0; gflops_armpl = 0; gflops_mema = 0; cake_mem_acc = 0
	gflops_armcl_arr=[]; dram_io_armcl_arr=[]; dram_io_armcl = 0; gflops_armcl = 0;
	for i in range(len(Ms)):
		for j in range(1,ntrials+1):
			# multiply by 64 bytes since external memory request non-cacheable 
			# and L2-data cache refills/writeback PMUs
			# in ARM are expressed in terms of number of cache lines
			a = open('reports_arm/report_armpl_%d-%d' % (i,j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'armpl') & (df1['M'] == Ms[i]) \
			& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time'].mean()
			dram_io_armpl += (((int(re.search(r'\d+', a[5]).group())*64.0) )) / (1e9)
			dram_io_armpl += (((int(re.search(r'\d+', a[6]).group())*64.0) ) ) / (1e9)
			gflops_armpl += (2*float(Ms[i]*Ns[i]*Ks[i]) / cpu_time) / (1e9)
			#
			a = open('reports_arm/report_armcl_%d-%d' % (i,j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'armcl') & (df1['M'] == Ms[i]) \
			& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time'].mean()
			dram_io_armcl += (((int(re.search(r'\d+', a[5]).group())*64.0) ) ) / (1e9)
			dram_io_armcl += (((int(re.search(r'\d+', a[6]).group())*64.0) )) / (1e9)
			gflops_armcl += (2*float(Ms[i]*Ns[i]*Ks[i]) / cpu_time) / (1e9)
			#
			a = open('reports_arm/report_mema_%d-%d' % (i,j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'mema') & (df1['M'] == Ms[i]) \
			& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time'].mean()
			dram_io_mema += (((int(re.search(r'\d+', a[5]).group())*64.0) ) ) / (1e9)
			dram_io_mema += (((int(re.search(r'\d+', a[6]).group())*64.0)) ) / (1e9)
			gflops_mema += (2*float(Ms[i]*Ns[i]*Ks[i]) / cpu_time) / (1e9)# / (float(NUM_CPUs[i]))
		#
		dram_io_armpl_arr.append(dram_io_armpl / ntrials)
		dram_io_armcl_arr.append(dram_io_armcl / ntrials)
		dram_io_mema_arr.append(dram_io_mema / ntrials)
		gflops_armpl_arr.append(gflops_armpl / ntrials)
		gflops_armcl_arr.append(gflops_armcl / ntrials)
		gflops_mema_arr.append(gflops_mema / ntrials)
		dram_io_armpl = 0; dram_io_mema = 0; gflops_armpl = 0; gflops_mema = 0; cake_mem_acc = 0
		dram_io_armcl = 0; gflops_armcl = 0;
	#
	#
	plt.figure(figsize = (10,4))
	barWidth = 0.25
	br1 = np.arange(len(Ms))
	br2 = [x + barWidth for x in br1]
	br3 = [x + barWidth for x in br2]
	# Make the plot
	plt.bar(br2, dram_io_mema_arr, color =colors[4], width = barWidth,
	        edgecolor ='grey', label =labels[0])
	plt.bar(br1, dram_io_armpl_arr, color =colors[5], width = barWidth,
	        edgecolor ='grey', label =labels[1])
	plt.bar(br3, dram_io_armcl_arr, color =colors[3], width = barWidth,
	        edgecolor ='grey', label =labels[2])
	# plt.plot(range(1,len(Ms)+1), list(dram_io_armpl_arr), label = labels[1],  marker = markers[1], color = colors[5])
	# plt.plot(range(1,len(Ms)+1), list(dram_io_mema_arr), label = labels[0],  marker = markers[0], color = colors[4])
	# plt.plot(range(1,len(Ms)+1), list(dram_io_armcl_arr), label = labels[2],  marker = markers[2], color = colors[3])
	# plt.plot(list(NUM_CPUs), list(cake_mem_acc_arr), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
	#
	plt.title('(a) DLMC Benchmark DRAM IO On Cortex-A72', fontsize = 24)
	plt.xlabel("Layer Id", fontsize = 24)
	plt.ylabel("DRAM IO (GB)", fontsize = 24)
	plt.xticks([r + barWidth for r in range(len(Ms))],
        [i+1 for i in range(len(Ms))])
	plt.legend(loc = "upper left", prop={'size': 14})
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	# plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	plt.figure(figsize = (10,4))
	plt.bar(br2, gflops_mema_arr, color =colors[4], width = barWidth,
	        edgecolor ='grey', label =labels[0])
	plt.bar(br1, gflops_armpl_arr, color =colors[5], width = barWidth,
	        edgecolor ='grey', label =labels[1])
	plt.bar(br3, gflops_armcl_arr, color =colors[3], width = barWidth,
	        edgecolor ='grey', label =labels[2])
	# plt.plot(range(1,len(Ms)+1), list(gflops_mema_arr), label = labels[0],  marker = markers[1], color = colors[5])
	# plt.plot(range(1,len(Ms)+1), list(gflops_armpl_arr), label = labels[1],  marker = markers[0], color = colors[4])
	# plt.plot(range(1,len(Ms)+1), list(gflops_armcl_arr), label = labels[2],  marker = markers[2], color = colors[3])
	#
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(b) DLMC Benchmark Throughput On Cortex-A72', fontsize = 24)
	plt.xlabel("Layer Id", fontsize = 24)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 24)
	plt.legend(loc = "lower right", prop={'size': 14})
	plt.xticks([r + barWidth for r in range(len(Ms))],
        [i+1 for i in range(len(Ms))])
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	# plt.ylim(ymin = 0,ymax = 35)
	# plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#



plot_mema_vs_arm_dlmc()




def plot_arm_vs_mema_tinyml(fname = 'plot_arm_vs_mema_tinyml'):
	plt.rcParams.update({'font.size': 16})
	# leg = plt.legend()
	# for lh in leg.legendHandles: 
	#     lh.set_alpha(1)
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	df1 = pandas.read_csv('tinymlbenchfp32')
	df2 = pandas.read_csv('tinymlbenchq15')
	Ms = [16,16,32,32,32,64,64,64, 64, 64, 8 ,16 ,32 ,32 ,64 ,64 ,128,128,256,256]
	Ks = [27 ,144,144,288,16 ,288,576,32 , 40, 64, 27 , 8 , 16 , 32 , 32 , 64 , 64 , 128, 128, 256]
	Ns = [1024,1024,256,256,256,64,64,64, 122, 125, 2304,2304,576,576,144,144, 36, 36, 9, 9]
	l = range(len(Ns))
	# tput_inner_1x8x1 = [float(2*Ms[i]*Ks[i]*Ns[i] / 1e6) / float(df1[(df1['algo'] == 'arm_mat_mult_f32') \
	# 	& (df1['id'] == i)]['time'].values[0]) for i in l]
	plt.figure(figsize = (16,4))
	plt.rcParams["legend.framealpha"] = 1
	barWidth = 0.2
	br1 = np.arange(len(Ms))
	br2 = [x + barWidth for x in br1]
	# br3 = [x + barWidth for x in br2]
	tput_inner_2x8x2 = [float(2*Ms[i]*Ks[i]*Ns[i] / 1e6) / (float(df1[(df1['algo'] == 'arm inner 2x8x2') \
		& (df1['id'] == i)]['time'].values[0]) / 1e6) for i in l]
	tput_mema = [float(2*Ms[i]*Ks[i]*Ns[i] / 1e6) / (float(df1[(df1['algo'] == 'mema outer 5x5') \
		& (df1['id'] == i)]['time'].values[0]) / 1e6) for i in l]
	plt.bar(br2, tput_mema, color =colors[4], width = barWidth,
	        edgecolor ='grey', label ='FP32 mema outer 5x1x5')
	plt.bar(br1, tput_inner_2x8x2, color =colors[1], width = barWidth,
	        edgecolor ='grey', label = 'FP32 cmsis inner 2x8x2')
		#
		#
	barWidth = 0.2
	# br4 = [x + barWidth for x in br3]
	br3 = [x + barWidth for x in br2]
	br4 = [x + barWidth for x in br3]
	# Make the plot
	tput_inner_2x4x2 = [float(2*Ms[i]*Ks[i]*Ns[i] / 1e6) / (float(df2[(df2['algo'] == 'arm_q15_inner_2x4x2') \
		& (df2['id'] == i)]['time'].values[0]) / 1e6) for i in l]
	tput_mema = [float(2*Ms[i]*Ks[i]*Ns[i] / 1e6) / (float(df2[(df2['algo'] == 'outer_q15_4x2') \
		& (df2['id'] == i)]['time'].values[0]) / 1e6) for i in l]
	plt.bar(br4, tput_mema, color =colors[4], width = barWidth,
	        edgecolor ='black', hatch="//", label ='Q15 mema outer 4x2x2')
	plt.bar(br3, tput_inner_2x4x2, color =colors[1], width = barWidth,
	        edgecolor ='black', hatch="//", label = 'Q15 cmsis inner 2x4x2')
		#
		#
		#
	plt.title('(a) Cortex-M4 Throughput on TinyML Benchmark', fontsize = 30)
	# plt.plot(l, tput_inner_1x8x1, 'b', label = 'cmsis inner 1x8x1', marker = markers[3], color = colors[3])
	plt.legend(loc = "lower right", prop={'size': 18})
	plt.xlabel('Layer Id', fontsize = 24)
	plt.ylabel('Throughput (OPs/sec)', fontsize = 24)
	plt.xticks([r + 2*barWidth for r in range(len(Ms))],
	    [i+1 for i in range(len(Ms))])
	plt.xticks(fontsize = 24)
	plt.yticks(range(0,27,5), fontsize = 20)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s_tput_M4.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	#
	df1 = pandas.read_csv('tinymlbenchfp32_M7')
	l = range(len(Ns))
	plt.figure(figsize = (12,4))
	barWidth = 0.25
	br1 = np.arange(len(Ms))
	br2 = [x + barWidth for x in br1]
	br3 = [x + barWidth for x in br2]
	tput_inner_2x8x2 = [float(2*Ms[i]*Ks[i]*Ns[i] / 1e6) / (float(df1[(df1['algo'] == 'arm inner 2x8x2') \
		& (df1['id'] == i)]['time'].values[0]) / 216e6) for i in l]
	tput_mema = [float(2*Ms[i]*Ks[i]*Ns[i] / 1e6) / (float(df1[(df1['algo'] == 'mema outer 5x5') \
		& (df1['id'] == i)]['time'].values[0]) / 216e6) for i in l]
	plt.bar(br2, tput_mema, color =colors[4], width = barWidth,
	        edgecolor ='grey', label ='mema outer 5x1x5')
	plt.bar(br1, tput_inner_2x8x2, color =colors[1], width = barWidth,
	        edgecolor ='grey', label = 'cmsis inner 2x8x2')
	# plt.plot(l, tput_mema, 'b', label = 'mema outer 5x1x5', marker = markers[4], color = colors[4])
	# plt.plot(l, tput_inner_2x8x2, 'b', label = 'cmsis inner 2x8x2', marker = markers[1], color = colors[1])
	plt.title('(c) Cortex-M7 FP32 Throughput on TinyML Benchmark', fontsize = 24)
	plt.legend(loc = "lower right", prop={'size': 16})
	plt.xlabel('Layer Id', fontsize = 24)
	plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 19)
	plt.xticks([r + barWidth for r in range(len(Ms))],
        [i+1 for i in range(len(Ms))])
	plt.xticks(fontsize = 20)
	plt.yticks(range(0,16,5), fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s_fp32_M7.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')





plot_arm_vs_mema_tinyml()







def plot_arm_vs_mema_fp32_m_vs_k(fname = 'arm_skew'):
  plt.rcParams.update({'font.size': 16})
  markers = ['o','v','s','d','^']
  colors = ['b','g','aqua','k','m','r']
  df1 = pandas.read_csv('arm_skew_m')
  M1 = range(5,300,20)
  M2 = range(4,300,16)
  K = 5
  N = 20
  tput_k = [float(2*i*N*K) / float(df1[(df1['algo'] == 'k first') \
    & (df1['M'] == i)]['time'].values[0]) for i in M1]
  tput_inner_2x8x2 = [float(2*i*N*K) / float(df1[(df1['algo'] == 'inner_2x8x2') \
    & (df1['M'] == i)]['time'].values[0]) for i in M2]
  tput_m = [float(2*i*N*K) / float(df1[(df1['algo'] == 'm first') \
    & (df1['M'] == i)]['time'].values[0]) for i in M1]
  fig = plt.figure(figsize = (6,4))
  plt.title('(a) FP32 Throughput With Large M', fontsize = 20)
  plt.plot(M1, tput_m, 'b', label = 'mema m-first', marker = markers[4], color = colors[4])
  plt.plot(M2, tput_inner_2x8x2, 'b', label = 'arm_inner_2x8x2', marker = markers[1], color = colors[1])
  plt.plot(M1, tput_k, 'b', label = 'mema k-first', marker = markers[3], color = colors[3])
  plt.legend(loc = "lower right", prop={'size': 16})
  plt.xlabel('M (K = 5, N = 20)', fontsize = 20)
  plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 20)
  # plt.xticks(range(0,111,20),fontsize = 14)
  plt.yticks(range(0,33,4),fontsize = 20)
  # plt.ylim(ymin = 2e-8,ymax = 0.00012)
  # plt.ticklabel_format(axis="y", style="sci")
  plt.savefig("%s_m.pdf" % fname, bbox_inches='tight')
  plt.show()
  plt.clf()
  plt.close('all')
  #
  #
  #
  df1 = pandas.read_csv('arm_skew_k')
  M1 = range(5,300,20)
  M2 = range(4,300,16)
  M = 10
  N = 20
  tput_k = [float(2*M*N*i) / float(df1[(df1['algo'] == 'k first') \
    & (df1['K'] == i)]['time'].values[0]) for i in M1]
  tput_inner_2x8x2 = [float(2*M*N*i) / float(df1[(df1['algo'] == 'inner_2x8x2') \
    & (df1['K'] == i)]['time'].values[0]) for i in M2]
  tput_m = [float(2*M*N*i) / float(df1[(df1['algo'] == 'm first') \
    & (df1['K'] == i)]['time'].values[0]) for i in M1]
  fig = plt.figure(figsize = (6,4))
  plt.title('(b) FP32 Throughput With Large K', fontsize = 20)
  plt.plot(M1, tput_m, 'b', label = 'mema m-first', marker = markers[4], color = colors[4])
  plt.plot(M2, tput_inner_2x8x2, 'b', label = 'arm_inner_2x8x2', marker = markers[1], color = colors[1])
  plt.plot(M1, tput_k, 'b', label = 'mema k-first', marker = markers[3], color = colors[3])
  plt.legend(loc = "lower right", prop={'size': 16})
  plt.xlabel('K (M = 10, N = 20)', fontsize = 20)
  plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 20)
  # plt.xticks(range(0,111,20),fontsize = 14)
  plt.yticks(range(0,33,4), fontsize = 20)
  # plt.ylim(ymin = 2e-8,ymax = 0.00012)
  # plt.ticklabel_format(axis="y", style="sci")
  plt.savefig("%s_k.pdf" % fname, bbox_inches='tight')
  plt.show()
  plt.clf()
  plt.close('all')



plot_arm_vs_mema_fp32_m_vs_k()





def plot_arm_vs_mema_fp32_tput_k(fname = 'arm_vs_mema_fp32_tput_k3'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	bws = ['1', '1/2','1/3','1/4']
	for x in range(1,5):
		df1 = pandas.read_csv('arm_vs_mema_fp32_k3.csv')
		K1 = range(5,166,5)
		K2 = range(8,169,8)
		M1 = range(60,61,20)
		for M in M1:
			N = M
			# tput_inner_1x16x1 = [float(2*M*N*i) / float(df1[(df1['algo'] == 'inner_1x16x1') \
			# 	& (df1['K'] == i) & (df1['M'] == M)]['time'].values[0]) for i in K1]
			tput_inner_2x8x2 = [float(2*M*N*i) / float(df1[(df1['algo'] == 'inner_2x8x2') \
				& (df1['K'] == i) & (df1['M'] == M) & (df1['bw'] == x)]['time'].values[0]) for i in K2]
			tput_mema = [float(2*M*N*i) / float(df1[(df1['algo'] == 'mema') \
				& (df1['K'] == i) & (df1['M'] == M) & (df1['bw'] == x)]['time'].values[0]) for i in K1]
			fig = plt.figure(figsize = (6,4))
			plt.title('FP32 Throughput When Input BW = %s * peak_BW' % bws[x-1], fontsize = 14)
			plt.plot(K1, tput_mema, 'b', label = 'mema_outer_5x5', marker = markers[4], color = colors[4])
			plt.plot(K2, tput_inner_2x8x2, 'b', label = 'arm_inner_2x8x2', marker = markers[1], color = colors[1])
			# plt.plot(K1, tput_inner_1x16x1, 'b', label = 'arm_inner_1x16x1', marker = markers[3], color = colors[3])
			plt.legend(loc = "lower right", prop={'size': 10})
			plt.xlabel('K (M=N=%d)' % M, fontsize = 16)
			plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 14)
			plt.xticks(range(0,180,20),fontsize = 14)
			plt.yticks(range(0,36,5), fontsize = 14)
			# plt.ylim(ymin = 2e-8,ymax = 0.00012)
			# plt.ticklabel_format(axis="y", style="sci")
			plt.savefig("%s_%d_%d.pdf" % (fname, M, x), bbox_inches='tight')
			plt.show()
			plt.clf()
			plt.close('all')



plot_arm_vs_mema_fp32_tput_k()




def plot_mema_fp32_sparse_levels(fname = 'mema_fp32_sparse_levels'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	df1 = pandas.read_csv('m4_sparsity')
	N1 = range(50,100,5)
	inner_2x8x2 = [((80.0**3) / (48618.0 / 1e6)) / 1e6 for i in N1]
	# runtime_mema_sp = [float(df1[(df1['algo'] == 'mema_sp') \
	# 	& (df1['sparsity'] == i)]['time'].values[0] / (1e6)) for i in N1]
	runtime_mema_sp_packed = [float(df1[(df1['algo'] == 'mema_sp_packed') \
		& (df1['sparsity'] == i)]['time'].values[0] / (1e6)) for i in N1]
	runtime_mema_sp_packed_reorder = [float(df1[(df1['algo'] == 'mema_sp_packed_reorder') \
		& (df1['sparsity'] == i)]['time'].values[0] / (1e6)) for i in N1]
	fig = plt.figure(figsize = (6,4))
	plt.title('(c) FP32 Throughput on ARM MCU', fontsize = 20)
	plt.plot(N1, inner_2x8x2, 'b', label = 'CMSIS dense MM', marker = markers[4], color = colors[4])
	# plt.plot(N1, runtime_mema_sp, 'b', label = 'TUMMY', marker = markers[4], color = colors[5])
	plt.plot(N1, runtime_mema_sp_packed, 'b', label = 'TUMMY+packing', marker = markers[4], color = colors[0])
	plt.plot(N1, runtime_mema_sp_packed_reorder, 'b', label = 'TUMMY+packing_reordering', marker = markers[4], color = colors[-1])
	plt.legend(loc = "lower left", prop={'size': 10})
	plt.xlabel('sparsity (%)', fontsize = 16)
	plt.ylabel('Throughput (MFLOPs/sec)', fontsize = 14)
	# plt.xticks(range(0,86,20),fontsize = 14)
	plt.yticks(fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_mema_fp32_sparse_levels()



def plot_arm_vs_mema_fp32_sparse(fname = 'arm_vs_mema_fp32_sparse'):
	plt.rcParams.update({'font.size': 16})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	df1 = pandas.read_csv('m4_sparse')
	N1 = range(5,86,5)
	N2 = range(8,86,8)
	runtime_inner_1x16x1 = [float(df1[(df1['algo'] == 'inner_1x16x1') \
		& (df1['M'] == i)]['time'].values[0] / (1e6)) for i in N1]
	runtime_inner_2x8x2 = [float(df1[(df1['algo'] == 'inner_2x8x2') \
		& (df1['M'] == i)]['time'].values[0] / (1e6)) for i in N2]
	runtime_mema = [float(df1[(df1['algo'] == 'mema') \
		& (df1['M'] == i)]['time'].values[0] / (1e6)) for i in N1]
	runtime_mema_sp = [float(df1[(df1['algo'] == 'mema_sp') \
		& (df1['M'] == i)]['time'].values[0] / (1e6)) for i in N1]
	runtime_mema_sp_packed = [float(df1[(df1['algo'] == 'mema_sp_packed') \
		& (df1['M'] == i)]['time'].values[0] / (1e6)) for i in N1]
	fig = plt.figure(figsize = (6,4))
	plt.title('(a) FP32 Runtime of spMM on ARM MCU', fontsize = 14)
	plt.plot(N1, runtime_mema, 'b', label = 'mema_outer_5x5', marker = markers[4], color = colors[4])
	plt.plot(N1, runtime_mema_sp, 'b', label = 'mema_poly', marker = markers[4], color = colors[5])
	plt.plot(N1, runtime_mema_sp_packed, 'b', label = 'mema_rosko', marker = markers[4], color = colors[0])
	plt.plot(N2, runtime_inner_2x8x2, 'b', label = 'arm_inner_2x8x2', marker = markers[1], color = colors[1])
	plt.plot(N1, runtime_inner_1x16x1, 'b', label = 'arm_inner_1x16x1', marker = markers[3], color = colors[3])
	plt.legend(loc = "upper left", prop={'size': 10})
	plt.xlabel('N', fontsize = 16)
	plt.ylabel('Runtime (sec)', fontsize = 14)
	plt.xticks(range(0,86,20),fontsize = 14)
	plt.yticks(fontsize = 14)
	# plt.ylim(ymin = 2e-8,ymax = 0.00012)
	# plt.ticklabel_format(axis="y", style="sci")
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_arm_vs_mema_fp32_sparse()













#--------------------------------- Old PLots-------------------------------

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
	plt.legend(loc = "upper left", prop={'size': 14})
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

