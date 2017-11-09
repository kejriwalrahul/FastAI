#include <stdio.h>
#include "../GameInterfaces/GameState.cu"
#include "../Includes/PriorityQueue.cu"
#include "../Includes/pq_kernels.cu"
#include <stdlib.h>

int main(){
	PriorityQueue* pq = new PriorityQueue();
	PriorityQueue* d_pq;
	InsertTable* h_instab, *d_instab;
	DeleteTable* h_deltab, *d_deltab;
	h_instab = new InsertTable();
	h_deltab = new DeleteTable();
	/*InsertTable *instab;
	cudaHostAlloc((void **)&instab,sizeof(InsertTable),0);
	instab = new InsertTable();*/
		
	cudaMalloc((void **)&d_pq,sizeof(PriorityQueue));
	cudaMemcpy(d_pq,pq,sizeof(PriorityQueue), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_instab,sizeof(InsertTable));
	cudaMalloc((void **)&d_deltab,sizeof(DeleteTable));
	cudaError_t err;
	int vals[3];
	int indices[1000];
	int num_indices;
	int *d_indices;
	cudaMalloc(&d_indices,1000*sizeof(int));
	for(int i=0;i<15;i++){
		//cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		for(int j=0;j<3;j++){
			vals[j] = rand()%100;
			printf("%d\n",vals[j]);
		}
		h_instab->addEntry(0,vals,3,i);
		//h_instab->printTable();
		cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
		
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_instab->status[j]==1 && h_instab->level[j]%2==0){
				indices[num_indices++] = j;
			}
		}
		cudaDeviceSynchronize();
		cudaMemcpy(d_indices,indices,num_indices*sizeof(int),cudaMemcpyHostToDevice);
		if(num_indices!=0)insert<<<1,num_indices>>>(d_pq,d_instab,d_indices,num_indices);
		
		cudaDeviceSynchronize();
		/*err = cudaGetLastError();
		printf("error=%d, %s, %s\n", err, cudaGetErrorName(err),cudaGetErrorString(err));*/
		cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_instab->status[j]==1 && h_instab->level[j]%2==1){
				indices[num_indices++] = j;
			}
		}
		if(num_indices!=0)insert<<<1,num_indices>>>(d_pq,d_instab,d_indices,num_indices);
		cudaMemcpy(h_instab,d_instab,sizeof(InsertTable), cudaMemcpyDeviceToHost);
		//h_instab->printTable();
		cudaMemcpy(pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
		h_instab->printTable();
		pq->print_object();
		printf("########################\n");
	}
	/*int vals[3] = {3,4,5};
	h_instab->addEntry(0,vals,3,0);
	cudaMemcpy(d_instab,h_instab,sizeof(InsertTable), cudaMemcpyHostToDevice);
	print_val<<<1,4>>>(d_pq);
	int off[1] = {0};
	int *d_off;
	cudaMalloc((void **)&d_off,sizeof(int));
	cudaMemcpy(d_off,off,sizeof(int),cudaMemcpyHostToDevice);
	writeToNode<<<1,4>>>(d_pq,d_instab,d_off,1);
	print_val<<<1,4>>>(d_pq);*/
	//print_val<<<1,1>>>(d_pq);
	cudaDeviceSynchronize();
	//h_instab->printTable();
	cudaMemcpy(pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
	pq->print_object();
	
	printf("HEllo %d\n",pq->curr_size);
	for(int j=0;j<3;j++){
			vals[j] = j*100;
			printf("%d\n",vals[j]);
		}
	vals[2] = INT_MAX;
	pq->deleteUpdate(vals,2,0);
	cudaMemcpy(d_pq,pq,sizeof(PriorityQueue), cudaMemcpyHostToDevice);
	h_deltab->addEntry();
	for(int i=0;i<2;i++){
		cudaMemcpy(d_deltab,h_deltab,sizeof(DeleteTable), cudaMemcpyHostToDevice);
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_deltab->status[j]==1 && h_deltab->level[j]%2==0){
				indices[num_indices++] = j;
			}
		}
		cudaMemcpy(d_indices,indices,num_indices*sizeof(int),cudaMemcpyHostToDevice);
		if(num_indices!=0)delete_update<<<1,num_indices>>>(d_pq,d_deltab,d_indices,num_indices);
		cudaMemcpy(pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
		pq->print_object();
		printf("########################\n");
		cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		num_indices = 0;
		for(int j=0;j<QSIZE;j++){
			if(h_deltab->status[j]==1 && h_deltab->level[j]%2==1){
				indices[num_indices++] = j;
			}
		}
		cudaMemcpy(d_indices,indices,num_indices*sizeof(int),cudaMemcpyHostToDevice);
		if(num_indices!=0)delete_update<<<1,num_indices>>>(d_pq,d_deltab,d_indices,num_indices);
		
		cudaMemcpy(h_deltab,d_deltab,sizeof(DeleteTable), cudaMemcpyDeviceToHost);
		cudaMemcpy(pq,d_pq,sizeof(PriorityQueue), cudaMemcpyDeviceToHost);
		pq->print_object();
		printf("########################\n");
	}
	
	return 0;
}
