#include <stdio.h>
#include "TicTacToe.h"

int main(){
	TicTacToeState *ttcs = new TicTacToeState();
	TicTacToeState *t1;
	printf("Original board\n");
	ttcs->print_board();
	t1 = ttcs->make_move(3);
	printf("First move\n");
	ttcs->print_board();
	printf("Second move\n");
	t1->make_move(4);
	t1->print_board();
	return 0;
}
