#include <stdio.h>
__global__ void print(PriorityQueue *pq){
	printf("%d %d\n",threadIdx.x,pq->getSize());
}
