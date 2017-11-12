#include <stdio.h>
#include "../GameInterfaces/Connect4.cu"
#include "../Includes/PriorityQueue_C4.cu"
#include "../Includes/kernels_c4.cu"
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include "../Includes/timer.h"
#define BRANCH_FACTOR_C4 20
#define NUM_CANDIDATES 1000


int main(){
	InsertTable *h_instab,*d_instab;
	DeleteTable *h_deltab,*d_deltab;
	int h_offsets[QSIZE],*d_offsets;
	Node h_to_insert[NUM_PER_NODE*BRANCH_FACTOR_C4];
	Node *d_to_insert;
	Node *d_to_send;
	Node *d_candidates;
	int *num_inserts;
	int *bestMove;
	PriorityQueue *h_pq,*d_pq;
	Connect4State *d_state;
	char board[BOARD_SIZE];
	int col[NUM_COLS];
	cudaMalloc((void **)&d_pq,sizeof(PriorityQueue));
	cudaMalloc((void **)&d_instab,sizeof(InsertTable));
	cudaMalloc((void **)&d_deltab,sizeof(DeleteTable));
	cudaMalloc((void **)&d_to_insert,NUM_PER_NODE*BRANCH_FACTOR_C4*sizeof(Node));
	cudaMalloc((void **)&d_to_send,NUM_PER_NODE*sizeof(Node));
	cudaMalloc((void **)&d_candidates,NUM_CANDIDATES*sizeof(Node));
	cudaMalloc((void **)&d_state,sizeof(Connect4State));
	cudaMalloc(&d_offsets,QSIZE*sizeof(int));
	cudaHostAlloc(&num_inserts,sizeof(int),0);
	cudaHostAlloc(&bestMove,sizeof(int),0);
	
	h_instab = new InsertTable();
	h_deltab = new DeleteTable();
	h_pq = new PriorityQueue();
	CPUTimer cputimer;
	
	
	bool isInsertDone;
	int insertedSize;
	int num_indices;
	bool player;
	player = false;
	cudaError_t err;
	int curr_size = 0;
	
	// Create root node
	
	//cudaMemcpy(d_state,h_state,sizeof(Connect4State),cudaMemcpyHostToDevice);
	int n,c,k;
	scanf("%d",&n);
	for(int i=0;i<BOARD_SIZE;i++){
		board[i] = '|';
	}
	for(int i=0;i<NUM_COLS;i++){
		col[i] = 0;
	}
	for(int i=0;i<n;i++){
		scanf("%d",&c);
		k = OFFSET(col[c],c);
		col[c]++;
		h_offsets[i] = k;
		if(i%2==0){
			board[k] = 'X';
		}
		else{
			board[k] = 'O';
		}
		player = !player;
	}
	printf("Initial Board\n");
	/*for(int i=0;i<NUM_ROWS;i++){
		for(int j=0;j<NUM_COLS;j++){
			printf("%c ",board[OFFSET(i,j)]);
		}
		printf("\n");
	}*/
	for(int i=NUM_ROWS-1; i>=0; i--){
		printf("|");
		for(int j=0; j<NUM_COLS; j++)
			printf("%c ", board[OFFSET(i,j)]);
		printf("\n");
	}
	
	cputimer.Start();
	cudaMemcpy(d_offsets,h_offsets,n*sizeof(int), cudaMemcpyHostToDevice);
	createRootNode<<<1,1>>>(d_to_insert,d_offsets,n);
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
	
	
	// At this stage all candidates are in the priority queue.
	bool *isEnd;
	cudaHostAlloc(&isEnd,sizeof(bool),0);
	*isEnd = false;
	
	*isEnd = false;
	int time = 0;
	int num_to_process,num_to_insert,num_to_send;
	int target;
	PQNode curr_root;
	
	*num_inserts = 0;
	int sum = 0;
	while(!(*isEnd)){
	//while(time < 20){
		cudaMemcpy(h_to_insert,d_to_insert,NUM_PER_NODE*BRANCH_FACTOR_C4*sizeof(Node), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		
		num_to_process = *num_inserts;
		
		curr_root = h_pq->readRoot();
		for(int i=0;i<curr_root.size;i++){
			h_to_insert[num_to_process++] = curr_root.nodes[i];
		}
		thrust::stable_sort(h_to_insert,h_to_insert+num_to_process);
		
		num_to_send = (num_to_process>NUM_TO_SEND)?NUM_TO_SEND:num_to_process;
		num_to_insert = num_to_process - num_to_send;
		//num_to_send = num_to_process-num_to_insert;
		
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
			h_instab->addEntry(0,h_to_insert+NUM_TO_SEND+NUM_PER_NODE+curr_size,num_to_insert,target);
			curr_size += insertedSize;
			num_to_insert -= insertedSize;
			target++;
		}
		cudaMemcpy(d_pq,h_pq,sizeof(PriorityQueue), cudaMemcpyHostToDevice);
		
		//cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		// Delete update on even level
		num_indices = 0;
		//h_deltab->printTable();
		for(int j=0;j<QSIZE;j++){
			if(h_deltab->status[j]==1 && h_deltab->level[j]%2==0){
				h_offsets[num_indices++] = j;
			}
		}
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_deltab,h_deltab,sizeof(DeleteTable), cudaMemcpyHostToDevice);
		//printf("even deletes: %d\n",num_indices);
		delete_update<<<(num_indices+1023/1024),1024>>>(d_pq,d_deltab,d_offsets,num_indices);
		
		cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		// Insert Update on even level
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_instab->status[j]==1 && h_instab->level[j]%2==0){
				h_offsets[num_indices++] = j;
			}
		}
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
		insert<<<(num_indices+1023/1024),1024>>>(d_pq,d_instab,d_offsets,num_indices);
		//printf("even inserts: %d\n",num_indices);
		cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		// Delete update on odd level
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_deltab->status[j]==1 && h_deltab->level[j]%2==1){
				h_offsets[num_indices++] = j;
			}
		}
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_deltab,h_deltab,sizeof(DeleteTable), cudaMemcpyHostToDevice);
		delete_update<<<(num_indices+1023/1024),1024>>>(d_pq,d_deltab,d_offsets,num_indices);
		//printf("odd deletes: %d\n",num_indices);
		cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		// Insert Update on odd level
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_instab->status[j]==1 && h_instab->level[j]%2==1){
				h_offsets[num_indices++] = j;
			}
		}
		cudaMemcpy(d_offsets,h_offsets,num_indices*sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
		insert<<<(num_indices+1023/1024),1024>>>(d_pq,d_instab,d_offsets,num_indices);
		//printf("odd inserts: %d\n",num_indices);
		// Call the SSS* application, TODO - put it in a different stream
		cudaMemcpy(d_to_send,h_to_insert,num_to_send*sizeof(Node),cudaMemcpyHostToDevice);
		sum += num_to_send;
		*num_inserts = 0;
		sss_star_algo<<<1,NUM_PER_NODE>>>(d_to_send,num_to_send,d_to_insert,num_inserts,isEnd,bestMove,player);
		cudaDeviceSynchronize();
		//err = cudaGetLastError();
		//printf("error=%d, %s, %s\n", err, cudaGetErrorName(err),cudaGetErrorString(err));
		//printf("### %d %d\n",*num_inserts,curr_root.size);
		time++;
	}
	
	//cudaDeviceSynchronize();
	//cudaMemcpy(h_pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
	cputimer.Stop();
	//printf("%d\n",*isEnd);
	//printf("%d\n",*bestMove);
	//printf("Iters: %d Nodes: %d\n",time,sum);
	if(n%2==0){
		board[OFFSET(col[*bestMove],*bestMove)] = 'X';
	}
	else{
		board[OFFSET(col[*bestMove],*bestMove)] = 'O';
	}
	printf("Final Board\n");
	for(int i=NUM_ROWS-1; i>=0; i--){
		printf("|");
		for(int j=0; j<NUM_COLS; j++)
			printf("%c ", board[OFFSET(i,j)]);
		printf("\n");
	}
	printf("Time taken: %lf milliseconds\n",cputimer.Elapsed()*1000);
	return 0;
}
