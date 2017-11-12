Some optimizations that were tried out:
	1) Using insertion sort instead of bubble sort(in the insert and delete kernels). It was found that bubble sort took comparatively lesser time than insertion sort. Also, the number of elements in one node of a priority queue is considerably small. For larger number of elements, we can use a kernel to sort the elements in parallel. In these demo cases, a kernel would have higher overheads.
	
	2) Performing operations in separate streams. The SSS* algorithm and the insert and delete updates of the priority queue can happen concurrently. When these are put in separate streams, we observe that the time taken is lesser.
	
	3) Writing the code in a way to reduce thread divergence also helped in speeding up the larger cases. For smaller cases, the time taken did not show much improvement. 

The optimizations have been done only for the Tic Tac Toe example. It can be extended to the Connect 4 example as well.
	
	Time taken for test case 6:
		a) Without optimizations: More than 9 seconds
		b) With streams optimization: About 7-8 seconds
		c) With streams and reducing thread divergence: About 6-7 seconds
		d) Without streams but with reduced thread divergence: About 8 seconds
		
Description of files:
	
	GameInterfaces/
		Connect4.cu - Connect4 class attributes and methods
		TicTacToe.cu - TicTacToe class attributes and methods
		GameState.cu - Base class for games
		
	GamePlayingAlgos/
		connect4.cu -  CUDA main program for connect 4 example
		tic_tac_toe.cu - CUDA main program for tic tac toe example
		opt_ttt.cu - Optimized code(with streams and reduction in divergence) for tic tac toe example
		script_ttt.sh - Script to test the cases of tic tac toe
		script_c4.sh - Script to test the cases of connect4
		
	Includes/
		kernels_(game) - Kernels for the corresponding game
		opt_kernel_ttt - Kernels with reduced thread divergence for tic tac toe
		PriorityQueue_(game) - Parallel Priority Queue class along with required data structures.
		timer.h - To time the program
		
	Tests/
		Basic test programs(irrelevant to the project)
		
