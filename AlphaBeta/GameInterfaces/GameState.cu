/*
	Generic Parent Class for all game interfaces

	Rahul Kejriwal
	CS14B023
*/

/*
	Abstract Class for abstracting actual game interface from game-playing algorithms
*/
class GameState {

public:

	/*
		Array to hold moves from current GameState
		Can be used to generate children 
	*/
	bool *moves;
	int moves_length;

	/*
		To store best known next move if computed
	*/
	int optimal_move;
	
	/*
		Store turn of player
			false = Player 0
			true  = Player 1
	*/
	bool turn;
	
	/*
		Stores the parent node and child number of parent which gives current node 
	*/
	GameState *parent; 
	GameState **children; 
	int child_num; 

	/*
		Stack vars for Alpha Beta
	*/
	int alpha;
	int beta;
	int depth;
	bool isMax;
	int next_move;
	int val;

	bool started_loop;
	int last_returned_val;
	GameState *prev;
	GameState *next;

	/*
		Evaluation function to be defined by concrete game interface
	*/
	__host__ __device__
	virtual int heuristicEval() = 0;

	/*
		Returns if the current game state is a terminal game tree node 
	*/
	__host__ __device__
	virtual bool isTerminal() = 0;

	/*
		Creates an array of possible moves in moves
	*/
	__host__ __device__
	virtual void moveGen() = 0;

	/*
		Returns the new game state after making the given move

		DANGER: No validity check for move # [Excersice Caution]
	*/
	__host__ __device__
	virtual GameState* makeMove(int) = 0;

	/*
		Prints Game Board for DEBUG purposes
	*/	
	__host__ __device__
	virtual void printState() = 0;
	
	
	__host__ __device__
	virtual int piece(int) = 0;


	/*
		Stack vars initialize
	*/
	__host__ __device__
	void stateReset(int a, int b, int d, bool iM){
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
