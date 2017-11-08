#include <iostream>
#include <TicTacToe.cu>

/*
	Recursive Definition
*/
int recursiveAlphaBeta(GameState *g, int alpha, int beta, int depth, bool isMax){
	bool done;
	int res = g->GoalCheck(done);
	if(depth == 0 || done)
		return res;

	bool *moves = g->MoveGen();
	int val;
	if(isMax){
		int temp = INT_MIN;
		for(int i=0; i<BOARD_SIZE; i++)
			if(moves[i]){
				val = max(val, recursiveAlphaBeta(g->make_move(i), alpha, beta, depth-1, !isMax));
				alpha = max(alpha, val); 
				if(beta <= alpha)	return val;
			}
	}
	else{
		int temp = INT_MAX;
		for(int i=0; i<BOARD_SIZE; i++)
			if(moves[i]){
				val = min(val, recursiveAlphaBeta(g->make_move(i), alpha, beta, depth-1, !isMax));
				beta = min(beta, val); 
				if(beta <= alpha)	return val;
			}
	}

	return val;
}

/*
	Iterative Definition
*/
int iterativeAlphaBeta(GameState *g, int alpha, int beta, int depth){
	
}