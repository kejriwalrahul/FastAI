/*
	Serial code for AlphaBeta

	Rahul Kejriwal
	CS14B023
*/

#include <climits>
#include <time.h>
#include <vector>
#include <stdio.h>
using namespace std;

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
__host__ __device__
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
	int count = 1;

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
			count++;
			continue;
		}
		// Else no more children
		else{
			returnHelper(state, state->val, fin_val);
			continue;
		}
	}

	printf("Nodes: %d\n", count);
	return fin_val;
}


/*
	Reorder move order
*/
__host__ __device__
void reorder(int *order, int *scores, bool isMax, int len){
	for(int k=0; k<len; k++)	order[k] = k;

	for(int i=0;i<len;i++)
		for(int j=0;j<len-i;j++){
			if((isMax && scores[j]<scores[j+1]) || (!isMax && scores[j]>scores[j+1])){
				int temp = scores[j];
				scores[j] = scores[j+1];
				scores[j+1] = temp;
				temp = order[j];
				order[j] = order[j+1];
				order[j+1] = temp;
			}
		}
}


/*
	Alpha beta, fixed depth
*/
__device__ 
int GPUIterativeAlphaBeta(GameState* state, int init_val, int *order){
	int fin_val = init_val;
	bool iM = !(state->turn);
	int len = state->moves_length;
	int *scores = new int[len];
	for(int i=0; i<len; i++)	scores[i] = 0;

	while(state){
		/*
			Phase 1 - Fresh Node discovered
		*/
		if(!(state->started_loop)){
			if(state->depth == 0 || state->isTerminal()){
				if(state->prev){
					// Transmit curr node info
					state->prev->last_returned_val = state->heuristicEval();				
					// Go up tree
					state = state->prev;
					// Collect garbage
					delete state->next;		
				}
				else{
					fin_val = state->heuristicEval();
					delete state;
					state = NULL;
				}
				continue;
			}

			state->moveGen();			
			state->started_loop = true;
		}

		/*
			Phase 2 - Update last return val (safe even for 1st time)
				Revisiting current node
		*/
		if(state->prev == NULL && state->next_move>=0)
			scores[order[state->next_move]] = state->last_returned_val;

		if(state->isMax){
			if(state->val < state->last_returned_val){
				state->val = state->last_returned_val;
				if(state->alpha < state->val)
					state->alpha = state->val;
			}
		}
		else{
			if(state->val > state->last_returned_val){
				state->val = state->last_returned_val;
				if(state->beta > state->val)
					state->beta = state->val;
			}
		}

		if(state->beta <= state->alpha){
			if(state->prev){
				// Transmit curr node info
				state->prev->last_returned_val = state->val;				
				// Go up tree
				state = state->prev;
				// Collect garbage
				delete state->next;		
			}
			else{
				fin_val = state->val;
				delete state;
				state = NULL;
			}
			continue;
		}

		state->next_move++;

		/*
			Phase 3 - Generate next child and start it
		*/

		// Find next valid move
		while(state->next_move < state->moves_length && !(state->moves[order[state->next_move]]))
			state->next_move++;

		// If child found
		if(state->next_move < state->moves_length){
			state->next = state->makeMove(order[state->next_move]);
			state->next->stateReset(state->alpha, state->beta, state->depth-1, !(state->isMax));
			state->next->prev = state;
			state = state->next;
			continue;
		}
		// Else no more children
		else{
			if(state->prev){
				// Transmit curr node info
				state->prev->last_returned_val = state->val;				
				// Go up tree
				state = state->prev;
				// Collect garbage
				delete state->next;		
			}
			else{
				fin_val = state->val;
				delete state;
				state = NULL;
			}
			continue;
		}
	}

	// reorder(order, scores, iM, len);
	return fin_val;
}


/*
	GPU kernel for Alpha Beta, Iterative deepening
*/
template<class gameTypeState>
__global__ void kernel(GameState **garr, int alpha, int beta, int depth, int isMax, int* res, int limit, unsigned int *once_done){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= limit)	return;
	
	res = res + idx;
	GameState *g = garr[idx];

	// printf("Thread %d: %d %d %d %d %d %d %d %d %d\n", idx, state->piece(0), state->piece(1), state->piece(2), state->piece(3), state->piece(4), state->piece(5), state->piece(6), state->piece(7), state->piece(8));

	int *order = new int[g->moves_length];
	for(int k=0;k<g->moves_length; k++)	order[k] = k;

	for(int d=2; d<=depth; d++){
		GameState *state = new gameTypeState(*(gameTypeState*)g);
		state->stateReset(alpha, beta, d, isMax);
		*res = GPUIterativeAlphaBeta(state, (isMax?INT_MIN:INT_MAX), order);	
		if(d==2)	atomicInc(once_done, INT_MAX);		
	}
}

template<class gameTypeState>
__global__ void kernel2(GameState **garr, int alpha, int beta, int depth, int isMax, int* res, int limit, unsigned int *once_done){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= limit)	return;
	
	res = res + idx;
	GameState *g = garr[idx];

	// printf("Thread %d: %d %d %d %d %d %d %d %d %d\n", idx, state->piece(0), state->piece(1), state->piece(2), state->piece(3), state->piece(4), state->piece(5), state->piece(6), state->piece(7), state->piece(8));

	int *order = new int[g->moves_length];
	for(int k=0;k<g->moves_length; k++)	order[k] = k;

	GameState *state = new gameTypeState(*(gameTypeState*)g);
	state->stateReset(alpha, beta, depth, isMax);
	*res = GPUIterativeAlphaBeta(state, (isMax?INT_MIN:INT_MAX), order);	
}


/*
	Accumulate children
*/
void add_children(GameState *g, int goto_depth, vector<GameState*> &children){
	
	// If leaf, add
	if(goto_depth == 0){
		children.push_back(g);
		return;
	}

	// Else, recurse till leaves
	g->moveGen();
	g->children = new GameState*[g->moves_length];
	g->started_loop = false; 
	for(int i=0; i<g->moves_length; i++)
		if(g->moves[i]){
			GameState *next_node = g->makeMove(i);
			add_children(next_node, goto_depth-1, children);
			g->children[i] = next_node;
		}
		else
			g->children[i] = NULL;
}


/*
	Computes minimax of root given kernel results
*/
int back_up(GameState *g, int goto_depth, int &best_move){
	if(goto_depth == 0)
		return g->val;

	best_move = 0;
	int temp_var;

	// Max node
	if(g->turn == false){
		g->val = INT_MIN;
		for(int i=0; i<g->moves_length; i++)
			if(g->moves[i]){
				int res = back_up(g->children[i], goto_depth-1, temp_var);
				if(g->val < res){
					g->val = res;
					best_move = i;
				}
			}
		return g->val;
	}
	else{
		g->val = INT_MAX;
		for(int i=0; i<g->moves_length; i++)
			if(g->moves[i]){
				int res = back_up(g->children[i], goto_depth-1, temp_var);
				if(g->val > res){
					g->val = res;
					best_move = i;
				}
			}
		return g->val;		
	}
}


/*
	Generate tree top and accumulate leaves
	Calls kernel on leaves
*/
template<class gameTypeState>
int cpu_alphabeta_starter(gameTypeState *g, int depth, int isMax, int time){
	int goto_depth = 3;
	vector<GameState*> children;

	/*
		Phase 1: Branch & Get children
	*/
	add_children(g, goto_depth, children);
	int num_leaves = children.size();

	GameState **arr;
	GameState **carr = new GameState*[num_leaves];
	cudaMalloc(&arr, num_leaves * sizeof(GameState*));
	for(int i=0; i<num_leaves; i++){
		GameState *node;
		cudaMalloc(&node, sizeof(gameTypeState));
		cudaMemcpy(node, children[i], sizeof(gameTypeState), cudaMemcpyHostToDevice);
		carr[i] = node;
	}
	cudaMemcpy(arr,carr,num_leaves*sizeof(GameState*), cudaMemcpyHostToDevice);

	/*
		Phase 2: Compute leaves using GPU
	*/

	int *res;
	unsigned int *once_done;
	cudaHostAlloc(&res, num_leaves * sizeof(int), 0);
	cudaHostAlloc(&once_done, sizeof(unsigned int), 0);

	// printf("Launch Config: %d, %d\n", 1+num_leaves/512, 512);
	if(time>0)
		kernel<gameTypeState><<<1+num_leaves/512, 512>>>(arr, INT_MIN, INT_MAX, depth-goto_depth, ((goto_depth%2)?!isMax:isMax), res, num_leaves, once_done);
	else
		kernel2<gameTypeState><<<1+num_leaves/512, 512>>>(arr, INT_MIN, INT_MAX, depth-goto_depth, ((goto_depth%2)?!isMax:isMax), res, num_leaves, once_done);
	
	int *fin_res = res;
	// int *fin_res = new int[num_leaves];
	// cudaMemcpy(fin_res, res, num_leaves * sizeof(int), cudaMemcpyDeviceToHost);

	/*
		Phase 3: Backup result
	*/
	int best_move;
	if(time>0){
		// printf("hello\n");
		while(*once_done < num_leaves){
			cudaError_t err = cudaGetLastError();  
			if (err != cudaSuccess){
				printf("Error: %s\n", cudaGetErrorString(err));
				exit(1);
			}
		};

		clock_t start;
		start = clock();
		while((clock()-start)/CLOCKS_PER_SEC<time){
			for(int i=0; i<num_leaves; i++)
				children[i]->val = fin_res[i];

			int fin_val = back_up(g, goto_depth, best_move);
			printf("%s %d\n", "Minimax value (GPU): ", fin_val);		
			
			clock_t delay_start;
			delay_start = clock();
			while((clock()-delay_start)/CLOCKS_PER_SEC<1);
		}
	}
	else{
		cudaDeviceSynchronize();	
		cudaError_t err = cudaGetLastError();  
		if (err != cudaSuccess)		printf("Error: %s\n", cudaGetErrorString(err));
	
		/*
			Check fin_res
		*/
		
		for(int i=0; i<num_leaves; i++){
			// printf("Thread %d: %d %d %d %d %d %d %d %d %d\n", i, children[i]->piece(0), children[i]->piece(1), children[i]->piece(2), children[i]->piece(3), children[i]->piece(4), children[i]->piece(5), children[i]->piece(6), children[i]->piece(7), children[i]->piece(8));

			int temp_res = recursiveAlphaBeta(children[i], INT_MIN, INT_MAX, depth-goto_depth, isMax);
			printf("Thread %d: %d %d\n",i, fin_res[i], temp_res);
		}
		
	
		for(int i=0; i<num_leaves; i++)
			children[i]->val = fin_res[i];

		int fin_val = back_up(g, goto_depth, best_move);
		printf("%s %d\n", "Minimax value (GPU): ", fin_val);		
	}

	return best_move;
}