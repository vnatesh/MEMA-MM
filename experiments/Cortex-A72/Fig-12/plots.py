import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys



	
# def plot_mema_vs_arm_dlmc(fname = 'mema_vs_arm_dlmc', ntrials = 10):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','aqua','k','m','r']
# 	labels = ['MEMA', 'ARMPL', 'ARMCL']
# 	Ms=(512,512,2048,512,512,2048,512,512,2048,512,512,2048)
# 	Ks=(2048,512,512,2048,512,512,2048,512,512,2048,512,512)
# 	Ns=(256,256,256,512,512,512,1024,1024,1024,2048,2048,2048)
# 	df1 = pandas.read_csv('results_dlmc')
# 	#
# 	#
# 	#
# 	gflops_armpl_arr=[];gflops_mema_arr=[];dram_io_mema_arr=[];dram_io_armpl_arr=[];cake_mem_acc_arr=[]
# 	dram_io_armpl = 0; dram_io_mema = 0; gflops_armpl = 0; gflops_mema = 0; cake_mem_acc = 0
# 	gflops_armcl_arr=[]; dram_io_armcl_arr=[]; dram_io_armcl = 0; gflops_armcl = 0;
# 	for i in range(len(Ms)):
# 		for j in range(1,ntrials+1):
# 			# multiply by 64 bytes since external memory request non-cacheable 
# 			# and L2-data cache refills/writeback PMUs
# 			# in ARM are expressed in terms of number of cache lines
# 			a = open('reports_arm/report_armpl_%d-%d' % (i,j),'r').read().split('\n')
# 			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
# 			cpu_time = df1[(df1['algo'] == 'armpl') & (df1['M'] == Ms[i]) \
# 			& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time'].mean()
# 			dram_io_armpl += (((int(re.search(r'\d+', a[5]).group())*64.0) )) / (1e9)
# 			dram_io_armpl += (((int(re.search(r'\d+', a[6]).group())*64.0) ) ) / (1e9)
# 			gflops_armpl += (2*float(Ms[i]*Ns[i]*Ks[i]) / cpu_time) / (1e9)
# 			#
# 			a = open('reports_arm/report_armcl_%d-%d' % (i,j),'r').read().split('\n')
# 			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
# 			cpu_time = df1[(df1['algo'] == 'armcl') & (df1['M'] == Ms[i]) \
# 			& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time'].mean()
# 			dram_io_armcl += (((int(re.search(r'\d+', a[5]).group())*64.0) ) ) / (1e9)
# 			dram_io_armcl += (((int(re.search(r'\d+', a[6]).group())*64.0) )) / (1e9)
# 			gflops_armcl += (2*float(Ms[i]*Ns[i]*Ks[i]) / cpu_time) / (1e9)
# 			#
# 			a = open('reports_arm/report_mema_%d-%d' % (i,j),'r').read().split('\n')
# 			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
# 			cpu_time = df1[(df1['algo'] == 'mema') & (df1['M'] == Ms[i]) \
# 			& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time'].mean()
# 			dram_io_mema += (((int(re.search(r'\d+', a[5]).group())*64.0) ) ) / (1e9)
# 			dram_io_mema += (((int(re.search(r'\d+', a[6]).group())*64.0)) ) / (1e9)
# 			gflops_mema += (2*float(Ms[i]*Ns[i]*Ks[i]) / cpu_time) / (1e9)# / (float(NUM_CPUs[i]))
# 		#
# 		dram_io_armpl_arr.append(dram_io_armpl / ntrials)
# 		dram_io_armcl_arr.append(dram_io_armcl / ntrials)
# 		dram_io_mema_arr.append(dram_io_mema / ntrials)
# 		gflops_armpl_arr.append(gflops_armpl / ntrials)
# 		gflops_armcl_arr.append(gflops_armcl / ntrials)
# 		gflops_mema_arr.append(gflops_mema / ntrials)
# 		dram_io_armpl = 0; dram_io_mema = 0; gflops_armpl = 0; gflops_mema = 0; cake_mem_acc = 0
# 		dram_io_armcl = 0; gflops_armcl = 0;
# 	#
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(range(1,len(Ms)+1), list(dram_io_armpl_arr), label = labels[1],  marker = markers[1], color = colors[5])
# 	plt.plot(range(1,len(Ms)+1), list(dram_io_mema_arr), label = labels[0],  marker = markers[0], color = colors[4])
# 	plt.plot(range(1,len(Ms)+1), list(dram_io_armcl_arr), label = labels[2],  marker = markers[2], color = colors[3])
# 	# plt.plot(list(NUM_CPUs), list(cake_mem_acc_arr), label = labels[2], color = colors[5], linewidth = 2, linestyle='dashed')
# 	#
# 	plt.title('(a) DLMC Benchmark DRAM IO On Cortex-A72')
# 	plt.xlabel("Problem ID", fontsize = 18)
# 	plt.ylabel("DRAM IO (GB)", fontsize = 18)
# 	plt.legend(loc = "upper left", prop={'size': 10})
# 	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
# 	plt.show()
# 	plt.clf()
# 	plt.close('all')
# 	#
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(range(1,len(Ms)+1), list(gflops_mema_arr), label = labels[0],  marker = markers[1], color = colors[5])
# 	plt.plot(range(1,len(Ms)+1), list(gflops_armpl_arr), label = labels[1],  marker = markers[0], color = colors[4])
# 	plt.plot(range(1,len(Ms)+1), list(gflops_armcl_arr), label = labels[2],  marker = markers[2], color = colors[3])
# 	#
# 	plt.ticklabel_format(useOffset=False, style='plain')
# 	plt.title('(b) DLMC Benchmark Throughput On Cortex-A72', fontsize = 18)
# 	plt.xlabel("Problem ID", fontsize = 18)
# 	plt.ylabel("Throughput (GFLOP/s)", fontsize = 18)
# 	plt.legend(loc = "lower right", prop={'size': 10})
# 	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
# 	plt.xticks(range(1,len(Ms)+1))
# 	plt.ylim(ymin = 0,ymax = 35)
# 	plt.show()
# 	plt.clf()
# 	plt.close('all')
# 	#



# plot_mema_vs_arm_dlmc()



# for i in range(len(Ms)):
# 	# multiply by 64 bytes since external memory request non-cacheable 
# 	# and L2-data cache refills/writeback PMUs
# 	# in ARM are expressed in terms of number of cache lines
# 	# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
# 	cpu_time = df1[(df1['algo'] == 'armpl') & (df1['M'] == Ms[i]) \
# 		& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time']
# 	print (cpu_time.std() / cpu_time.mean())*100
# 	cpu_time = df1[(df1['algo'] == 'mema') & (df1['M'] == Ms[i]) \
# 		& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time']
# 	print (cpu_time.std() / cpu_time.mean())*100
# 	cpu_time = df1[(df1['algo'] == 'armcl') & (df1['M'] == Ms[i]) \
# 		& (df1['K'] == Ks[i]) & (df1['N'] == Ns[i])]['time']
# 	print (cpu_time.std() / cpu_time.mean())*100


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
	plt.xlabel("MxKxN Dimensions of Layer", fontsize = 24)
	plt.ylabel("DRAM IO (GB)", fontsize = 24)
	plt.xticks([r + barWidth for r in range(len(Ms))],
        ["%dx%dx%d" % (Ms[i],Ks[i],Ns[i]) for i in range(len(Ms))])
	plt.legend(loc = "upper left", prop={'size': 14})
	plt.xticks(rotation=60, fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
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
	plt.xlabel("MxKxN Dimensions of Layer", fontsize = 24)
	plt.ylabel("Throughput (GFLOP/s)", fontsize = 24)
	plt.legend(loc = "lower right", prop={'size': 14})
	plt.xticks([r + barWidth for r in range(len(Ms))],
        ["%dx%dx%d" % (Ms[i],Ks[i],Ns[i]) for i in range(len(Ms))], fontsize = 12)
	plt.xticks(rotation=60, fontsize = 20)
	plt.yticks(fontsize = 20)
	# plt.ylim(ymin = 0,ymax = 35)
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#



plot_mema_vs_arm_dlmc()



def plot_cake_vs_arm_cpu(M,N,K,mc,kc,alpha,fname = 'cake_vs_arm', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['MEMA Observed', 'ARMPL Observed','MEMA Optimal',
	'MEMA extrapolated', 'ARMPL extrapolated','ARMCL Observed', 
	'ARMCL extrapolated']
	NUM_CPUs = [1,2,3,4]
	gflops_armpl_arr=[];gflops_mema_arr=[];dram_bw_mema_arr=[];dram_bw_armpl_arr=[];cake_mem_acc_arr=[]
	dram_bw_armpl = 0; dram_bw_mema = 0; gflops_armpl = 0; gflops_mema = 0; cake_mem_acc = 0
	gflops_armpl_arr=[]; dram_bw_armpl_arr=[]; dram_bw_armcl = 0; gflops_armcl = 0;
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
			dram_bw_armpl += (((int(re.search(r'\d+', a[5]).group())*64.0) ) / cpu_time) / (1e9)
			dram_bw_armpl += (((int(re.search(r'\d+', a[6]).group())*64.0) ) / cpu_time) / (1e9)
			gflops_armpl += (2*float(M*N*K) / cpu_time) / (1e9)
			#
			a = open('reports_arm/report_armcl_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'armcl') & (df1['p'] == NUM_CPUs[i])]['time'].mean()
			dram_bw_armcl += (((int(re.search(r'\d+', a[5]).group())*64.0) ) / cpu_time) / (1e9)
			dram_bw_armcl += (((int(re.search(r'\d+', a[6]).group())*64.0) ) / cpu_time) / (1e9)
			gflops_armcl += (2*float(M*N*K) / cpu_time) / (1e9)
			#
			a = open('reports_arm/report_cake_%d-%d' % (NUM_CPUs[i],j),'r').read().split('\n')
			# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
			cpu_time = df1[(df1['algo'] == 'cake') & (df1['p'] == NUM_CPUs[i])]['time'].mean()
			dram_bw_mema += (((int(re.search(r'\d+', a[5]).group())*64.0) ) / cpu_time) / (1e9)
			dram_bw_mema += (((int(re.search(r'\d+', a[6]).group())*64.0)) / cpu_time) / (1e9)
			gflops_mema += (2*float(M*N*K) / cpu_time) / (1e9)# / (float(NUM_CPUs[i]))
		#
		dram_bw_armpl_arr.append(dram_bw_armpl / ntrials)
		dram_bw_armpl_arr.append(dram_bw_armcl / ntrials)
		dram_bw_mema_arr.append(dram_bw_mema / ntrials)
		gflops_armpl_arr.append(gflops_armpl / ntrials)
		gflops_armpl_arr.append(gflops_armcl / ntrials)
		gflops_mema_arr.append(gflops_mema / ntrials)
		dram_bw_armpl = 0; dram_bw_mema = 0; gflops_armpl = 0; gflops_mema = 0; cake_mem_acc = 0
		dram_bw_armcl = 0; gflops_armcl = 0;
	#
	# plt.subplot(1, 2, 1)
	plt.figure(figsize = (6,4))
	plt.plot(list(NUM_CPUs), list(dram_bw_armpl_arr), label = labels[1],  marker = markers[1], color = colors[5])
	plt.plot(list(NUM_CPUs), list(dram_bw_mema_arr), label = labels[0],  marker = markers[0], color = colors[4])
	plt.plot(list(NUM_CPUs), list(dram_bw_armpl_arr), label = labels[5],  marker = markers[2], color = colors[3])
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
	y = [gflops_armpl_arr[-2] + (gflops_armpl_arr[-1] - gflops_armpl_arr[-2])*i - 0.006*i*i for i in range(4)]
	y += 2*[y[-1]]
	plt.plot(x, y, color = colors[5], linestyle = 'dashed', label = labels[4])
	#
	x = np.array(list(range(3,9)))
	y = [gflops_armpl_arr[-2] + (gflops_armpl_arr[-1] - gflops_armpl_arr[-2])*i - 0.006*i*i for i in range(4)]
	y += 2*[y[-1]]
	plt.plot(x, y, color = colors[3], linestyle = 'dashed', label = labels[6])
	#
	plt.plot(list(range(1,9)), [gflops_mema_arr[0]+i*(gflops_mema_arr[0]) for i in range(8)], 
		label = labels[3], linewidth = 2, linestyle = 'dashed', color = colors[4])
	plt.xticks(list(range(1,9)))
	#
	plt.plot(list(NUM_CPUs), list(gflops_armpl_arr), label = labels[1],  marker = markers[1], color = colors[5])
	plt.plot(list(NUM_CPUs), list(gflops_mema_arr), label = labels[0],  marker = markers[0], color = colors[4])
	plt.plot(list(NUM_CPUs), list(gflops_armpl_arr), label = labels[5],  marker = markers[2], color = colors[3])
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
