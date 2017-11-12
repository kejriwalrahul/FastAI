#!/bin/bash

for i in $(seq 1 1 4)
do
	echo $i
	./connect4.out < cases_c4/tc$i.txt > outfile.out
	diff outfile.out cases_c4/tc$i.out.txt
	echo "###"
done
