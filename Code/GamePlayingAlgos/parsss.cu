#include <stdio.h>
#include "../GameInterfaces/TicTacToe.cu"
#include "../Includes/PriorityQueue.cu"
#include "../Includes/pq_kernels.cu"
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#define BRANCH_FACTOR_TIC 10

int main(){
	InsertTable *h_instab,*d_instab;
	DeleteTable *h_deltab,*d_deltab;
	int h_offsets[QSIZE],*d_offsets;
	Node h_to_insert[NUM_PER_NODE*BRANCH_FACTOR_TIC];
	Node *d_to_insert;
	int *num_inserts;
	PriorityQueue *h_pq,*d_pq;
	
	cudaMalloc((void **)&d_pq,sizeof(PriorityQueue));
	cudaMalloc((void **)&d_instab,sizeof(InsertTable));
	cudaMalloc((void **)&d_deltab,sizeof(DeleteTable));
	cudaMalloc((void **)&d_to_insert,NUM_PER_NODE*BRANCH_FACTOR_TIC*sizeof(Node));
	cudaMalloc(&d_offsets,QSIZE*sizeof(int));
	cudaHostAlloc(&num_inserts,sizeof(int),0);
	
	h_instab = new InsertTable();
	h_deltab = new DeleteTable();
	h_pq = new PriorityQueue();
	
	Node node_list[2*NUM_PER_NODE];
	bool isInsertDone;
	int insertedSize;
	int num_indices;
	cudaError_t err;
	
	// Create root node
	Node root(INT_MAX-1,new TicTacToeState());
	node_list[0] = root;
	insertedSize = 0;
	int curr_size = 0;
	do{
		
		h_instab->addEntry(0,node_list+curr_size,1,h_pq->getInsertTarget(1,&isInsertDone,&insertedSize));
		curr_size += insertedSize;
	}while(!isInsertDone);
	cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
	cudaMemcpy(d_deltab,h_deltab,sizeof(DeleteTable), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pq,h_pq,sizeof(PriorityQueue), cudaMemcpyHostToDevice);
	
	num_indices = 1;
	h_offsets[0] = 0;
	cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
	insert<<<1,1>>>(d_pq,d_instab,d_offsets,num_indices,num_inserts);
	cudaDeviceSynchronize();
	cudaMemcpy(h_pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
	
	
	// At this stage the root node is present in the priority queue.
	
	bool isEnd = false;
	int time = 0;
	int num_to_process,num_to_insert,num_to_send;
	PQNode curr_root;
	
	*num_inserts = 0;
	while(time<10){
		cudaMemcpy(h_to_insert,d_to_insert,NUM_PER_NODE*BRANCH_FACTOR_TIC*sizeof(Node), cudaMemcpyDeviceToHost);
		num_to_process = *num_inserts;
		curr_root = h_pq->readRoot();
		for(int i=0;i<curr_root.size;i++){
			h_to_insert[num_to_process++] = curr_root.nodes[i];
		}
		thrust::sort(h_to_insert,h_to_insert+num_to_process);
		num_to_insert = (num_to_process>NUM_PER_NODE)?num_to_process-NUM_PER_NODE:0;
		num_to_send = num_to_process-num_to_insert;
		h_pq->deleteUpdate(h_to_insert+num_to_send,num_to_insert,0);
		
		time++;
	}
	
	
	
	
	return 0;
}
