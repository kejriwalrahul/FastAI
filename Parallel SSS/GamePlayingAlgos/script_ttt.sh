#!/bin/bash

for i in $(seq 1 1 6)
do
	echo $i
	./tic_tac_toe.out < cases_ttt/tc$i.txt > outfile.out
	diff outfile.out cases_ttt/tc$i.out.txt
	echo "###"
done
