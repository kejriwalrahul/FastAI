 /*
	Game Interface for Tic-Tac-Toe

	Rahul Kejriwal
	CS14B023
*/

#include <stdio.h>
#include "GameState.cu"

#define BOARD_SIZE 9
#define WIN_SIZE 8
#define ROW_SIZE 3
#define NUM_ROWS 3
#define NUM_COLS 3

#define OFFSET(i,j) ((i)*NUM_COLS + (j))

__device__
int GPU_winning_patterns[WIN_SIZE][ROW_SIZE] = {
	{0, 1, 2},
	{3, 4, 5},
	{6, 7, 8},
	{0, 3, 6},
	{1, 4, 7},
	{2, 5, 8},
	{0, 4, 8},
	{2, 4, 6},
};

int CPU_winning_patterns[WIN_SIZE][ROW_SIZE] = {
	{0, 1, 2},
	{3, 4, 5},
	{6, 7, 8},
	{0, 3, 6},
	{1, 4, 7},
	{2, 5, 8},
	{0, 4, 8},
	{2, 4, 6},
};

class TicTacToeState : public GameState {

	/*
		2 arrays for game board:  
			occupied - whether position is occupied
			owner    - if occupied, which player's piece is it
	
		Linearized arrays
			i <-> (i/3, i%3)
	*/
	bool occupied[BOARD_SIZE];
	bool owner[BOARD_SIZE];

	/*
		Store turn of player
			false = Player 0
			true  = Player 1
	*/
	bool turn;

	/*
		Store if game is over
		If over, store winner
	*/
	bool isOver;
	int  winner;
	
	bool isSolved;
	bool isRoot;

public:
	TicTacToeState *parent_node;
	int bestMove;
	/*	
		Initialize game state
	*/
	__host__ __device__
	TicTacToeState(){
		for(int i=0; i<BOARD_SIZE; i++)
			occupied[i] = false;
		turn = false;
		parent_node = NULL;
		isSolved = false;
		isRoot = false;
	}
	
	
	/*
		Populates moves of parent with possible moves		
	*/
	 __host__ __device__		
	void moveGen(){
		moves_length = BOARD_SIZE;
		moves = new bool[BOARD_SIZE];
		for(int i=0; i<BOARD_SIZE; i++)
			moves[i] = !occupied[i];
	}
	
	 __host__ __device__		
	void moveGen(int *num_moves){
		*num_moves = 0;
		moves_length = BOARD_SIZE;
		moves = new bool[BOARD_SIZE];
		for(int i=0; i<BOARD_SIZE; i++){
			moves[i] = !occupied[i];
			*num_moves += moves[i];
		}
	}


	/*
		Returns if the current game state is a terminal game tree node 
	*/
	__host__ __device__
	bool isTerminal(){
		return isOver;
	}


	/*
		Evaluation function to be defined by concrete game interface
	*/
	__host__ __device__
	int heuristicEval(bool player){
		if(player){
			return winner;
		}
		return -winner;
	}
	__host__ __device__
	int heuristicEval(){
		
		return -winner;
	}


	/*
		Updates isOver and winner
	*/
	__host__ __device__
	void updateIfWinner(){
		isOver = false;
		
		// Check if somebody won
		for(int i=0; i<WIN_SIZE; i++){
			#ifdef  __CUDA_ARCH__
			if(
				occupied[GPU_winning_patterns[i][0]] &&
				occupied[GPU_winning_patterns[i][1]] &&
				occupied[GPU_winning_patterns[i][2]] &&
				owner[GPU_winning_patterns[i][0]] == owner[GPU_winning_patterns[i][1]] && 
				owner[GPU_winning_patterns[i][1]] == owner[GPU_winning_patterns[i][2]] 
			){
				isOver = true;
				winner = (owner[GPU_winning_patterns[i][0]] == false)? -1 : 1;
				return;
			}
			#else
			if(
				occupied[CPU_winning_patterns[i][0]] &&
				occupied[CPU_winning_patterns[i][1]] &&
				occupied[CPU_winning_patterns[i][2]] &&
				owner[CPU_winning_patterns[i][0]] == owner[CPU_winning_patterns[i][1]] && 
				owner[CPU_winning_patterns[i][1]] == owner[CPU_winning_patterns[i][2]] 
			){
				isOver = true;
				winner = (owner[CPU_winning_patterns[i][0]] == false)? -1 : 1;
				return;
			}
			#endif
		}

		// Check if game was draw
		bool done = true;
		for(int i=0; i<BOARD_SIZE; i++)
			done &= occupied[i];
		if(done){
			isOver = true;
			winner = 0;
			return;
		}

		// Game unfinished
		winner = 0;
		return;
	}

	/*
		Creates new TicTacToeState by making move at loc
		
		Note: Does not check validity of move, assumes it is correct
	*/
	 __host__ __device__
	TicTacToeState* makeMove(int loc){
		TicTacToeState *new_state = new TicTacToeState(*this);
		new_state->turn = !this->turn;		
		new_state->occupied[loc] = true; 
		new_state->owner[loc] = this->turn;
		new_state->parent_node = this;
		int child_val = loc;
		/*for(int i=0;i<=loc;i++){
			//if(this->occupied[i]==0){
				child_val++;
			//}
		}*/
		new_state->child_num = child_val;
		new_state->updateIfWinner();
		return new_state;
	}
	

	/*
		Prints the board for DEBUG purposes
	*/
	__host__ __device__
	void printState(){
		for(int i=0; i<NUM_ROWS; i++){
			for(int j=0; j<NUM_ROWS; j++)
				printf(occupied[OFFSET(i,j)]?(owner[OFFSET(i,j)]?"O ":"X "):"- "); 
			printf("\n");
		}
	}
	
	/*
		isLastChild for SSS*
	*/
	
	__host__ __device__
	bool isLastChild(){
		for(int i=BOARD_SIZE-1;i>=0;i--){
			if((parent_node)->occupied[i]==0&&occupied[i]==1){
				return true;
			}
			else if((parent_node)->occupied[i]==0){
				return false;
			}	
		}
		return false;
	}
	
	__host__ __device__
	int getNextChild(){
		int index;
		int flag = 0;
		for(int i=0;i<BOARD_SIZE;i++){
			if(flag == 0){
				if(!((parent_node)->occupied[i]) && occupied[i]){
					index = i;
					flag = 1;
					//printf("in next child %d\n",index);				
				}
			}
			else{
				if(!((parent_node)->occupied[i])){
					index = i;
					break;
				}
			}
		}
		return index;
		
	}
	
	__host__ __device__
	bool getSolved(){
		return isSolved;
	}
	
	__host__ __device__
	void setSolved(bool a){
		isSolved = a;
	}
	
	__host__ __device__
	bool getRoot(){
		return isRoot;
	}
	
	__host__ __device__
	void setRoot(bool a){
		isRoot = a;
	}
	
	__host__ __device__
	bool getOver(){
		return isOver;
	}
	
	__host__ __device__
	bool getTurn(){
		return turn;
	}
	
	__host__ __device__
	void setTurn(bool a){
		this->turn = a;
	}
};
