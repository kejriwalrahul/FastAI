 /*
	Game Interface for Tic-Tac-Toe

	Rahul Kejriwal
	CS14B023
*/

#define BOARD_SIZE 9
#define ROW_SIZE 3
#define WIN_SIZE 8

int winning_patterns[WIN_SIZE][ROW_SIZE] = {
	{0, 1, 2},
	{3, 4, 5},
	{6, 7, 8},
	{0, 3, 6},
	{1, 4, 7},
	{2, 5, 8},
	{0, 4, 8},
	{2, 4, 6},
};

class TicTacToeState {

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

public:

	/*	
		Initialize game state
	*/
	TicTacToeState(){
		for(int i=0; i<BOARD_SIZE; i++)
			occupied[i] = false;
		turn = false;
	}

	/*
		Returns list of possible moves
		
		Returns:
			bool[BOARD_SIZE] - ith element is true if that move is possible
	*/
	bool* MoveGen(){
		bool *moves = new bool[BOARD_SIZE];
		for(int i=0; i<BOARD_SIZE; i++)
			moves[i] = !occupied[i];
		return moves;
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
	int GoalCheck(bool &finished){
		finished = false;
		
		// Check if somebody won
		for(int i=0; i<WIN_SIZE; i++){
			if(
				occupied[winning_patterns[i][0]] &&
				occupied[winning_patterns[i][1]] &&
				occupied[winning_patterns[i][2]] &&
				owner[winning_patterns[i][0]] == owner[winning_patterns[i][1]] && 
				owner[winning_patterns[i][1]] == owner[winning_patterns[i][2]] 
			){
				finished = true;
				return (owner[winning_patterns[i][0]] == false)? -1 : 1;
			}
		}

		// Check if game was draw
		bool done = true;
		for(int i=0; i<BOARD_SIZE; i++)
			done &= occupied[i];
		if(done){
			finished = true;
			return 0;
		}

		// Game unfinished
		return 0;
	}

	/*
		Creates new TicTacToeState by making move at loc
		
		Note: Does not check validity of move, assumes it is correct
	*/
	TicTacToeState* make_move(int loc){
		TicTacToeState *new_state = new TicTacToeState(*this);
		new_state->turn = !this->turn;
		occupied[loc] = true; 
		owner[loc] = this->turn;
		return new_state;
	}

};