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
	TicTacToeState *parent_node;

public:

	/*	
		Initialize game state
	*/
	__host__ __device__
	TicTacToeState(){
		for(int i=0; i<BOARD_SIZE; i++)
			occupied[i] = false;
		turn = false;
	}


	/*
		Copy Constructor
	*/	
	__host__ __device__
	TicTacToeState(TicTacToeState &other){
		isOver = other.isOver;
		winner = other.winner;
		turn   = other.turn;
		moves_length = other.moves_length;

		memcpy(owner, other.owner, BOARD_SIZE*sizeof(bool));
		memcpy(occupied, other.occupied, BOARD_SIZE*sizeof(bool));
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
		int child_val = 0;
		for(int i=0;i<=loc;i++){
			if(this->occupied[i]==0){
				child_val++;
			}
		}
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
		int last_empty = BOARD_SIZE;
		for(int i=BOARD_SIZE-1;i>=0;i--){
			if((parent_node)->occupied[i]==0){
				last_empty = i;
				break;
			}
		}
		if(last_empty == child_num){
			return true;
		}
		return false;
	}

};
