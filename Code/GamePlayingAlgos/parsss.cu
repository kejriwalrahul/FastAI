#include <stdio.h>
#include "../GameInterfaces/TicTacToe.cu"
#include "../Includes/PriorityQueue.cu"
#include "../Includes/pq_kernels.cu"
#include <thrust/host_vector.h>

int main(){
	InsertTable *instab;
	DeleteTable *deltab;
	PriorityQueue *pq;
	
	cudaHostAlloc((void **)&pq,sizeof(PriorityQueue),0);
	cudaHostAlloc((void **)&instab,sizeof(InsertTable),0);
	cudaHostAlloc((void **)&deltab,sizeof(DeleteTable),0);
	
	
	return 0;
}
