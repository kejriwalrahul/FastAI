#!/bin/bash

for i in $(seq 1 1 6)
do
	echo $i
	./parsss.out < testcases/tc$i.txt > outfile.out
	diff outfile.out testcases/tc$i.out.txt
	echo "###"
done
