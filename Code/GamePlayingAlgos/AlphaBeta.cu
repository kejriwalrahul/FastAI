/*
	Serial code for AlphaBeta

	Rahul Kejriwal
	CS14B023
*/

#include <climits>

/*
	Recursive Definition
*/
int recursiveAlphaBeta(GameState *g, int alpha, int beta, int depth, bool isMax){
	
	// If reached depth limit or a leaf node in game tree
	if(depth == 0 || g->isTerminal())
		return g->heuristicEval();

	// Populate the moves for current state
	g->moveGen();

	int val;
	if(isMax){
		val = INT_MIN;
		for(int i=0; i<g->moves_length; i++)
			if(g->moves[i]){
				val = max(val, recursiveAlphaBeta(g->makeMove(i), alpha, beta, depth-1, false));
				alpha = max(alpha, val); 

				// Beta Cut-off
				if(beta <= alpha)	return val;
			}
	}
	else{
		val = INT_MAX;
		for(int i=0; i<g->moves_length; i++)
			if(g->moves[i]){
				val = min(val, recursiveAlphaBeta(g->makeMove(i), alpha, beta, depth-1, true));
				beta = min(beta, val);

				// Alpha Cut-off
				if(beta <= alpha)	return val;
			}
	}

	return val;
}

/*
	Iterative Definition
*/
int iterativeAlphaBeta(GameState *g, int alpha, int beta, int depth){
	
	// TO DO ... 

	return 0;
}