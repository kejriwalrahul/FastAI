 /*
	Game Interface for Connect-4

	Rahul Kejriwal
	CS14B023
*/

#include <stdio.h>
#include "GameState.cu"

#define NUM_ROWS 6
#define NUM_COLS 7
#define BOARD_SIZE (NUM_ROWS*NUM_COLS)

#define OFFSET(i,j) ((i)*NUM_COLS + (j))

__device__
int GPU_evaluationTable[NUM_ROWS][NUM_COLS] = {
	{3, 4, 5, 7, 5, 4, 3}, 
	{4, 6, 8, 10, 8, 6, 4},
	{5, 8, 11, 13, 11, 8, 5}, 
	{5, 8, 11, 13, 11, 8, 5},
	{4, 6, 8, 10, 8, 6, 4},
	{3, 4, 5, 7, 5, 4, 3}
};

int CPU_evaluationTable[NUM_ROWS][NUM_COLS] = {
	{3, 4, 5, 7, 5, 4, 3}, 
	{4, 6, 8, 10, 8, 6, 4},
	{5, 8, 11, 13, 11, 8, 5}, 
	{5, 8, 11, 13, 11, 8, 5},
	{4, 6, 8, 10, 8, 6, 4},
	{3, 4, 5, 7, 5, 4, 3}
};

class Connect4State : public GameState {

	/*
		2 arrays for game board:  
			occupied - whether position is occupied
			owner    - if occupied, which player's piece is it
	
		Linearized arrays
			i <-> (i/NUM_COLS, i%NUM_COLS)
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

	/*
		Store current heuristic evaluation of the game-state

		Board eval = p0_hval - p1_hval
	*/
	short p0_hval;
	short p1_hval;

public:

	/*	
		Initialize game state
	*/

	__host__ __device__
	Connect4State(){
		turn   = false;
		isOver = false;
		winner = 0;
		for(int i=0; i<BOARD_SIZE; i++)		
			occupied[i] = false;
		p0_hval = p1_hval = 0;
	}	


	/*
		Creates an array of possible moves in moves
	*/
	__host__ __device__		
	void moveGen(){
		moves_length = NUM_COLS;
		moves = new bool[NUM_COLS];
		for(int i=0; i<NUM_COLS; i++)
			moves[i] = !occupied[OFFSET(NUM_ROWS-1, i)];
	}


	/*
		Check if game is over and who won

		Returns:
			-1: Player 0 won
			 0: Game Draw
			 1: Player 1 won

			Sets finished if game is over else finished is false
			[Interpret return value only if game is over]
	*/
	__host__ __device__
	int GoalCheck(bool &finished){
		finished = isOver;
		return winner;
	}


	/*
		Returns if the current game state is a terminal game tree node 
	*/
	__host__ __device__
	bool isTerminal(){
		return isOver;
	}


	/*
		Creates new TicTacToeState by making move at loc
		
		Note: Does not check validity of move, assumes it is correct
	*/
	__host__ __device__
	Connect4State* makeMove(int loc){

		// Create new board and modify turn
		Connect4State *new_state = new Connect4State(*this);
		new_state->turn = !this->turn;

		// Update Board Status
		int row;
		for(row=0; row<NUM_ROWS; row++)
			if(!occupied[OFFSET(row, loc)])
				break;

		new_state->occupied[OFFSET(row, loc)] = true;		
		new_state->owner[OFFSET(row, loc)]    = this->turn;		

		// Update if game is over - by virtue of no empty place
		new_state->isOver = new_state->occupied[OFFSET(NUM_ROWS-1, 0)];
		for(int i=1; i<NUM_COLS; i++)
			new_state->isOver &= new_state->occupied[OFFSET(NUM_ROWS-1, i)];

		// Check if current move gave a win to current player
		new_state->update_win(row, loc);

		// Update heuristic values
		if(this->turn == false){
			#ifdef  __CUDA_ARCH__
			p0_hval += GPU_evaluationTable[row][loc];
			#else
			p0_hval += CPU_evaluationTable[row][loc];
			#endif			
		}
		else{
			#ifdef  __CUDA_ARCH__
			p1_hval += GPU_evaluationTable[row][loc];
			#else
			p1_hval += CPU_evaluationTable[row][loc];
			#endif			
		}

		return new_state;
	}


	/*
		Updates isOver and winner after a move
	*/
	__host__ __device__
	void update_win(int row, int loc){
		short l = 0, r = 0, t = 0, b = 0;
		short tl = 0, tr = 0, bl = 0, br = 0; 
		short dl = 1, dr = 1, dt = 1, db = 1;
		short dtl = 1, dtr = 1, dbl = 1, dbr = 1; 
		
		bool my_turn = !this->turn;

		for(int i=1; i<=3; i++){
			// Update left
			if(loc-i>=0 && occupied[OFFSET(row, loc-i)] && owner[OFFSET(row, loc-i)] == my_turn)		
				l+= dl;
			else	dl = 0;
			
			// Update right
			if(loc+i<NUM_COLS && occupied[OFFSET(row, loc+i)] && owner[OFFSET(row, loc+i)] == my_turn)	
				r+= dr;
			else	dr = 0;

			// Update top
			if(row+i<NUM_ROWS && occupied[OFFSET(row+i, loc)] && owner[OFFSET(row+i, loc)] == my_turn)	
				t+= dt;
			else	
				dt = 0;

			// Update bottom
			if(row-i>=0 && occupied[OFFSET(row-i, loc)] && owner[OFFSET(row-i, loc)] == my_turn)		
				b+= db;
			else	
				db = 0;

			// Update top-left
			if(loc-i>=0 && row+i<NUM_ROWS && occupied[OFFSET(row+i, loc-i)] && owner[OFFSET(row+i, loc-i)] == my_turn)			
				tl+= dtl;
			else	
				dtl = 0;
			
			// Update top-right
			if(loc+i<NUM_COLS && row+i<NUM_ROWS && occupied[OFFSET(row+i, loc+i)] && owner[OFFSET(row+i, loc+i)] == my_turn)	
				tr+= dtr;
			else	
				dtr = 0;

			// Update bottom-left
			if(loc-i>=0 && row-i>=0 && occupied[OFFSET(row-i, loc-i)] && owner[OFFSET(row-i, loc-i)] == my_turn)				
				bl+= dbl;
			else	
				dbl = 0;

			// Update bottom-right
			if(loc+i<NUM_COLS && row-i>=0 && occupied[OFFSET(row-i, loc+i)] && owner[OFFSET(row-i, loc+i)] == my_turn)			
				br+= dbr;
			else	
				dbr = 0;			
		}

		if((l+r+1>=4) || (t+b+1>=4) || (tl+br+1>=4) || (tr+bl+1>=4)){
			isOver = true;
			winner = my_turn;
		}
	}


	/*
		Returns heuristic evaluation of the gamestate from p0 perspective
	*/
	__host__ __device__
	int heuristicEval(){
		return p0_hval - p1_hval;
	}


	/*
		Print GameState
	*/
	__host__ __device__
	void printState(){
		printf("Next turn: %s\n", (turn?"P1":"P0")); 
		for(int i=NUM_ROWS-1; i>=0; i--){
			printf("|");
			for(int j=0; j<NUM_COLS; j++)
				printf("%s%s", occupied[OFFSET(i,j)]?(owner[OFFSET(i,j)]?"O":"X"):" ", "|");
			printf("\n");
		}
	}

};