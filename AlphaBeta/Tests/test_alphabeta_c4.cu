/*
	Test to check functioning of serial code of Alpha-Beta pruning

	Rahul Kejriwal
	CS14B023
*/

#include <iostream>
#include <climits>
#include "../GameInterfaces/Connect4.cu"
#include "../GamePlayingAlgos/AlphaBeta.cu"
using namespace std;

int main(){
	Connect4State *g = new Connect4State;

	// cout << "Minimax value: " << iterativeAlphaBeta(g, INT_MIN, INT_MAX, 9, false) << endl;

	/*
	cout << endl << "On GPU iterative version: " << endl;

	TicTacToeState *dg, *cg = new TicTacToeState;
	cg = cg->makeMove(0);
	
	int *res, ans;
	cudaMalloc(&dg, sizeof(TicTacToeState));
	cudaMalloc(&res, sizeof(int));
	cudaMemcpy(dg, cg, sizeof(TicTacToeState), cudaMemcpyHostToDevice);

	kernel<<<1,1>>>(dg, INT_MIN, INT_MAX, 9, false, res);
	cudaDeviceSynchronize();	
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess)		printf("Error: %s\n", cudaGetErrorString(err));
	else						printf("No Error\n");

	cudaMemcpy(&ans, res, sizeof(int), cudaMemcpyDeviceToHost);
	*/

	int depth = 7;
	cout << "Minimax value (CPU): " << recursiveAlphaBeta(g, INT_MIN, INT_MAX, depth, !(g->turn)) << endl;
	cout << "Best Move (GPU): " << cpu_alphabeta_starter<Connect4State>(g, depth, !(g->turn), -1) << endl;
	return 0;
}