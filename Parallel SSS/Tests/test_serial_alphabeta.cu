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

	cout << "Minimax value: " << recursiveAlphaBeta(g, INT_MIN, INT_MAX, 9, true) << endl;

	return 0;
}