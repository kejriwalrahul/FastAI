/*
	Test to check functioning of serial code of Alpha-Beta pruning

	Rahul Kejriwal
	CS14B023
*/

#include <iostream>
#include <climits>
#include "../GameInterfaces/TicTacToe.cu"
#include "../GamePlayingAlgos/AlphaBeta.cu"
using namespace std;

int main(){
	GameState *g = new TicTacToeState;

	// g = g->makeMove(0);
	// g = g->makeMove(1);
	// g = g->makeMove(2);
	// g = g->makeMove(3);
	// g = g->makeMove(4);
	// g = g->makeMove(5);
	// g = g->makeMove(7);
	// g = g->makeMove(6);

	cout << "Minimax value: " << recursiveAlphaBeta(g, INT_MIN, INT_MAX, 9, true) << endl;
	cout << "Minimax value: " << iterativeAlphaBeta(g, INT_MIN, INT_MAX, 9, true) << endl;

	cout << endl << "On GPU iterative version: " << endl;

	TicTacToeState *dg, *cg = new TicTacToeState;
	int *res, ans;
	cudaMalloc(&dg, sizeof(TicTacToeState));
	cudaMalloc(&res, sizeof(int));
	cudaMemcpy(dg, cg, sizeof(TicTacToeState), cudaMemcpyHostToDevice);

	kernel<<<1,1>>>(dg, INT_MIN, INT_MAX, 9, true, res);
	cudaDeviceSynchronize();	
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess)		printf("Error: %s\n", cudaGetErrorString(err));
	else						printf("No Error\n");

	cudaMemcpy(&ans, res, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Minimax value: " << ans << endl;
	return 0;
}