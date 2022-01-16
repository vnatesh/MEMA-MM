#!/bin/bash


# compile cake_sgemm_test
make;

NTRIALS=5;
# NCORES=4;

# run matmul bench
for ((j=1; j <= $NTRIALS; j++));
do
	for i in {500..10000..500}
	do
		./cake_sgemm_test $i ;
	done
done




# python3 plots.py $NTRIALS; 
