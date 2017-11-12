/*
	Test to check basic functioning of Tic Tac Toe interface

	Rahul Kejriwal
	CS14B023
*/

#include <iostream>
#include "../GameInterfaces/TicTacToe.cu"
using namespace std;

__global__ void test_kernel(){
	GameState *initial_stage = new TicTacToeState;
	initial_stage = initial_stage->makeMove(0);
	initial_stage = initial_stage->makeMove(4);
	initial_stage = initial_stage->makeMove(8);
	initial_stage->printState();
}

void test_function(){
	GameState *initial_stage = new TicTacToeState;
	initial_stage = initial_stage->makeMove(0);
	initial_stage = initial_stage->makeMove(4);
	initial_stage = initial_stage->makeMove(8);
	initial_stage->printState();	
}

int main(){

	cout << "Checking TicTacToe interface" << endl 
		 << "----------------------------" << endl;

	cout << "Using Kernel: " << endl;
	test_kernel<<<1,1>>>();
	cudaDeviceSynchronize();

	cout << "Using function: " << endl;
	test_function();

	cout << endl;

	return 0;
}