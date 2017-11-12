#!/bin/bash

for i in $(seq 1 1 4)
do
	echo Doing Testcase $i
	./c4_out <Testcases/Connect4/tc$i.txt > Outputs/out_$i.out
done
