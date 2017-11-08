/*
	Generic Parent Class for all game interfaces

	Rahul Kejriwal
	CS14B023
*/

/*
	Abstract Class for abstracting actual game interface from game-playing algorithms
*/
class GameState {

protected:

	/*
		Array to hold moves from current GameState
		Can be used to generate children 
	*/
	bool *moves;
	int moves_length;

public:

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
	
};