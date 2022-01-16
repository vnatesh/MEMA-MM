# Overview
This repository contains outer-product-based matrix multiplication kernels for ARM Cortex-M4 microcontrollers such as Arduino Nano 33 BLE.

## Installation

Before cloning and running the MEMA-MM sketch, please download and install the folowing dependencies:

* `CAKE_on_CPU (https://github.com/vnatesh/CAKE_on_CPU/)`
* `Arduino IDE`



## Quick Start

The current Arduino sketch (MEMA-MM.ino) contains example programs running MM using different inner and outer product kernels.

## Running Experiments:

Before running experiments, make sure the following additional dependencies are installed:


* `Cortex-M4` 
	* `Mbed OS 6.15.1` 
	* `ARM CMSIS-DSP Library version 5.8.0` 

* `Cortex-A72` 
	* `ARMPL 21.0.0` 
	* `ARM Compute Library (ARMCL) version 21.11`
	* `OpenMP 4.5` 
	* `perf version 5.4.157` 


* `Intel-i9 10900K`
	* `OpenMP 4.5` 


The experiments are organized in separate directories for each CPU architecture tested. Each arch-specific directory contains sub-directories corresponding to figures. To run an experiment and plot the associated figure, simply enter the directory and execute the `run.sh` file. An example to generate Figure 12 for the Cortex-A72 CPU tested is shown below. Experiments should be performed in `sudo` mode to enable permissions for hardware profiling.

```bash
~/MEMA-MM$ cd experiments/Cortex-A72/Fig-12
~/MEMA-MM/experiments/Cortex-A72/Fig-12$ ./run.sh
```


<!-- <p align = "center">
<img  src="https://github.com/vnatesh/maestro/blob/master/images/cake_diagram.png" width="500">
</p>
 -->


