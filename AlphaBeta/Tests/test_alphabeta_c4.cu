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

	int num_moves;
	cin >> num_moves; 

	for(int i=0; i<num_moves; i++){
		int move;
		cin >> move;
		g = g->makeMove(move);
	}

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
	int best_move = cpu_alphabeta_starter<Connect4State>(g, depth, !(g->turn), -1, 3);
	cout << "Best Move (GPU): " << best_move << endl;
	
	cout << "Initial Board: " << endl;
	g->printState();

	cout << "Next Board: " << endl;
	g->makeMove(best_move)->printState();

	return 0;
}