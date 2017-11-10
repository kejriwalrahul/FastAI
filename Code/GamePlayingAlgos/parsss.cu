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
	Node *d_to_send;
	int *num_inserts;
	PriorityQueue *h_pq,*d_pq;
	TicTacToeState *h_state,*d_state;
	
	cudaMalloc((void **)&d_pq,sizeof(PriorityQueue));
	cudaMalloc((void **)&d_instab,sizeof(InsertTable));
	cudaMalloc((void **)&d_deltab,sizeof(DeleteTable));
	cudaMalloc((void **)&d_to_insert,NUM_PER_NODE*BRANCH_FACTOR_TIC*sizeof(Node));
	cudaMalloc((void **)&d_to_send,NUM_PER_NODE*sizeof(Node));
	cudaMalloc((void **)&d_state,sizeof(TicTacToeState));
	cudaMalloc(&d_offsets,QSIZE*sizeof(int));
	cudaHostAlloc(&num_inserts,sizeof(int),0);
	
	h_instab = new InsertTable();
	h_deltab = new DeleteTable();
	h_pq = new PriorityQueue();
	
	Node node_list[2*NUM_PER_NODE];
	bool isInsertDone;
	int insertedSize;
	int num_indices;
	cudaError_t err;int curr_size = 0;
	
	// Create root node
	/*Node root(INT_MAX-1,new TicTacToeState());
	node_list[0] = root;
	
	cudaMemcpy(h_pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);*/
	
	//cudaMemcpy(d_state,h_state,sizeof(TicTacToeState),cudaMemcpyHostToDevice);
	h_offsets[0] = 0;
	h_offsets[1] = 3;
	//h_offsets[2] = 1;
	//h_offsets[3] = 4;
	cudaMemcpy(d_offsets,h_offsets,2*sizeof(int), cudaMemcpyHostToDevice);
	createRootNode<<<1,1>>>(d_to_insert,d_offsets,2);
	cudaMemcpy(h_to_insert,d_to_insert,sizeof(Node), cudaMemcpyDeviceToHost);
	insertedSize = 0;	
	do{
		
		h_instab->addEntry(0,h_to_insert+curr_size,1,h_pq->getInsertTarget(1,&isInsertDone,&insertedSize));
		curr_size += insertedSize;
	}while(!isInsertDone);
	cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
	cudaMemcpy(d_deltab,h_deltab,sizeof(DeleteTable), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pq,h_pq,sizeof(PriorityQueue), cudaMemcpyHostToDevice);
	
	num_indices = 1;
	h_offsets[0] = 0;
	cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
	insert<<<1,1>>>(d_pq,d_instab,d_offsets,num_indices);
	cudaDeviceSynchronize();
	cudaMemcpy(h_pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
	
	
	// At this stage the root node is present in the priority queue.
	
	bool *isEnd;
	cudaHostAlloc(&isEnd,sizeof(bool),0);
	*isEnd = false;
	int time = 0;
	int num_to_process,num_to_insert,num_to_send;
	int target;
	PQNode curr_root;
	
	*num_inserts = 0;
	while(!(*isEnd)){
		cudaMemcpy(h_to_insert,d_to_insert,NUM_PER_NODE*BRANCH_FACTOR_TIC*sizeof(Node), cudaMemcpyDeviceToHost);
		h_pq->print_object();
		cudaMemcpy(h_pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		
		num_to_process = *num_inserts;
		curr_root = h_pq->readRoot();
		for(int i=0;i<curr_root.size;i++){
			h_to_insert[num_to_process++] = curr_root.nodes[i];
		}
		thrust::sort(h_to_insert,h_to_insert+num_to_process);
		
		num_to_insert = (num_to_process>NUM_PER_NODE)?num_to_process-NUM_PER_NODE:0;
		num_to_send = num_to_process-num_to_insert;
		//printf("%d %d %d Num processed\n",num_to_process,num_to_insert,num_to_send);
		h_pq->deleteUpdate(h_to_insert+num_to_send,num_to_insert,0);
		if(num_to_insert>0){
			h_deltab->addEntry();
		}
		num_to_insert -= NUM_PER_NODE;
		
		// Add the remaining to insert update.
		isInsertDone = false;
		curr_size = 0;
		insertedSize = 0;
		target = h_pq->getInsertTarget(num_to_insert,&isInsertDone,&insertedSize);
		while(num_to_insert>0){			
			h_instab->addEntry(0,h_to_insert+2*NUM_PER_NODE+curr_size,num_to_insert,target);
			curr_size += insertedSize;
			num_to_insert -= insertedSize;
			target++;
		}
		cudaMemcpy(d_pq,h_pq,sizeof(PriorityQueue), cudaMemcpyHostToDevice);
		
		cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		// Delete update on even level
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_deltab->status[j]==1 && h_deltab->level[j]%2==0){
				h_offsets[num_indices++] = j;
			}
		}
		//printf("%d num deletes at even level\n",num_indices);
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_deltab,h_deltab,sizeof(DeleteTable), cudaMemcpyHostToDevice);
		delete_update<<<(num_indices+1023/1024),1024>>>(d_pq,d_deltab,d_offsets,num_indices);
		
		//cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		// Insert Update on even level
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_instab->status[j]==1 && h_instab->level[j]%2==0){
				h_offsets[num_indices++] = j;
			}
		}
		//printf("%d num inserts at even level\n",num_indices);
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
		insert<<<(num_indices+1023/1024),1024>>>(d_pq,d_instab,d_offsets,num_indices);
		
		//cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		
		cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		// Delete update on odd level
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_deltab->status[j]==1 && h_deltab->level[j]%2==1){
				h_offsets[num_indices++] = j;
			}
		}
		//printf("%d num deletes at odd level\n",num_indices);
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_deltab,h_deltab,sizeof(DeleteTable), cudaMemcpyHostToDevice);
		delete_update<<<(num_indices+1023/1024),1024>>>(d_pq,d_deltab,d_offsets,num_indices);
		
		cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		// Insert Update on odd level
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_instab->status[j]==1 && h_instab->level[j]%2==1){
				h_offsets[num_indices++] = j;
			}
		}
		//printf("%d num inserts at odd level\n",num_indices);
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
		insert<<<(num_indices+1023/1024),1024>>>(d_pq,d_instab,d_offsets,num_indices);
		
		// Call the SSS* application, TODO - put it in a different stream
		cudaMemcpy(d_to_send,h_to_insert,num_to_send*sizeof(Node),cudaMemcpyHostToDevice);
		*num_inserts = 0;
		sss_star_algo<<<1,NUM_PER_NODE>>>(d_to_send,num_to_send,d_to_insert,num_inserts,isEnd);
		cudaDeviceSynchronize();
		//printf("Num to insert: %d\n",*num_inserts);
		time++;
	}
	
	cudaDeviceSynchronize();
	cudaMemcpy(h_pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
	//h_pq->print_object();
	//h_instab->printTable();
	//printf("%d\n",*isEnd);
	
	return 0;
}
