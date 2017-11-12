#!/bin/bash

for i in $(seq 1 1 7)
do
	echo Doing Testcase $i
	./ttt_out <Testcases/TicTacToe/tc$i.txt > Outputs/out_$i.out
done