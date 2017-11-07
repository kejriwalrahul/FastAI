#include "PriorityQueue.h"
#include "pq_kernels.h"


int main(){
	PriorityQueue* pq = new PriorityQueue();
	PriorityQueue* d_pq;
	
	cudaMalloc((void **)&d_pq,sizeof(PriorityQueue));
	cudaMemcpy(d_pq,pq,sizeof(PriorityQueue), cudaMemcpyHostToDevice);
	print<<<1,32>>>(d_pq);
	cudaDeviceSynchronize();
	return 0;
}
