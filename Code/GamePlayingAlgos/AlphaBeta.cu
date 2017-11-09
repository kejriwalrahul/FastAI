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

class AlphaBetaState {

public:
	/*
		State Info
	*/
	GameState *gs;
	int alpha;
	int beta;
	int depth;
	bool isMax;
	int next_move;
	int val;

	bool started_loop;
	int last_returned_val;
	AlphaBetaState *prev;
	AlphaBetaState *next;

	/*
		Construct and initialize
	*/	
	__host__ __device__
	AlphaBetaState(GameState *g, int a, int b, int d, bool iM){
		gs    = g;
		alpha = a;
		beta  = b;
		depth = d;
		isMax = iM;
		next_move = -1;
		prev = next = NULL;
		if(iM)	last_returned_val = val = INT_MIN;
		else	last_returned_val = val = INT_MAX;
		started_loop = false;
	}
};

/*
	Helper return routine for iterative AlphaBeta
*/
inline void returnHelper(AlphaBetaState*& state, int val, int& fin_val){
	if(state->prev){
		// Transmit curr node info
		state->prev->last_returned_val = val;				
		// Go up tree
		state = state->prev;
		// Collect garbage
		delete state->next;		
	}
	else{
		fin_val = val;
		delete state;
		state = NULL;
	}
}

/*
	Iterative Definition
*/
int iterativeAlphaBeta(GameState *g, int alpha, int beta, int depth, int isMax){
	
	// Create root
	AlphaBetaState *state = new AlphaBetaState(g, alpha, beta, depth, isMax);
	int fin_val = (isMax?INT_MIN:INT_MAX);

	while(state){

		/*
			Phase 1 - Fresh Node discovered
		*/
		if(!(state->started_loop)){
			if(state->depth == 0 || state->gs->isTerminal()){
				returnHelper(state, state->gs->heuristicEval(), fin_val);
				continue;
			}

			state->gs->moveGen();			
			state->started_loop = true;
		}

		/*
			Phase 2 - Update last return val (safe even for 1st time)
				Revisiting current node
		*/
		if(state->isMax){
			state->val = max(state->val, state->last_returned_val);
			state->alpha = max(state->alpha, state->val);
		}
		else{
			state->val = min(state->val, state->last_returned_val);
			state->beta = min(state->beta, state->val);
		}

		if(state->beta <= state->alpha){
			returnHelper(state, state->val, fin_val);
			continue;
		}

		state->next_move++;

		/*
			Phase 3 - Generate next child and start it
		*/

		// Find next valid move
		while(state->next_move < state->gs->moves_length && !(state->gs->moves[state->next_move]))
			state->next_move++;

		// If child found
		if(state->next_move < state->gs->moves_length){
			state->next = new AlphaBetaState(state->gs->makeMove(state->next_move), state->alpha, state->beta, state->depth-1, !(state->isMax));
			state->next->prev = state;
			state = state->next;
			continue;
		}
		// Else no more children
		else{
			returnHelper(state, state->val, fin_val);
			continue;
		}
	}

	return fin_val;
}