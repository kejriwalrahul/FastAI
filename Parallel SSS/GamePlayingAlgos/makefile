tic_tac_toe: tic_tac_toe.cu
	nvcc tic_tac_toe.cu -o tic_tac_toe.out
	
opt_tic_tac_toe: opt_ttt.cu
	nvcc opt_ttt.cu -o tic_tac_toe.out
	
connect4: connect4.cu
	nvcc connect4.cu -o connect4.out

clean: 
	rm -rf *.o *.out
