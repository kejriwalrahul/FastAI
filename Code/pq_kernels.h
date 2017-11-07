#include <stdio.h>

__global__ void print_val(PriorityQueue *pq)
{
	printf("%d %d\n",threadIdx.x,pq->getSize());
}

__global__ void insert(PriorityQueue *pq, InsertTable *table,int* offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	if(index < size){
		int offset = offsets[index];
		if(table->indices[offset] == table->target[offset]){
			
			
			pq->writeToNode(table->elements[offset],table->num_elements[offset],table->target[offset]);
			
			table->status[offset] = 0;
			table->num_elements[offset] = 0;
		}
		else{
			int arr[2*R];
			int tot_size = 0;
			int tmp;
			for(int i=0;i<table->num_elements[offset];i++){
				arr[tot_size++] = table->elements[offset][i];
			}
			int node_num = table->indices[offset];
			for(int i=0;i<pq->nodes[node_num].size;i++){
				arr[tot_size++] = pq->nodes[node_num].nodes[i].key;
			}
			for(int i=0;i<tot_size;i++){
				for(int j=i+1;j<tot_size;j++){
					if(arr[i] > arr[j]){
						tmp = arr[i];
						arr[i] = arr[j];
						arr[j] = tmp;
					}
				}
			}
			for(int i=0;i<R&&i<tot_size;i++){
				pq->nodes[node_num].nodes[i].key = arr[i];
			}
			if(tot_size>R){
				for(int i=R;i<tot_size;i++){
					table->elements[offset][i-R] = arr[i];
				}
			}
			// Change the entry so that it moves towards the target.
			int level = table->level[offset];
			int target = table->target[offset];
			int rem;
			rem = table->target_bits[offset][level];
			table->indices[offset] = (node_num)*2+1+rem;
			table->level[offset] += 1;
			/*if(table->target[offset]==8){
				printf("Hello ");
				for(int i=0;i<10;i++){
					printf("%d ",table->target_bits[offset][i]);
				}
				printf("\n");
			}
			if(table->target[offset]==4)printf(" Hello %d %d %d %d\n",node_num,(node_num)*2+1+rem,table->target[offset],level);*/
			table->num_elements[offset] = (tot_size>R)?tot_size-R:0;
		}
	}
}

__global__ void writeToNode(PriorityQueue* pq, InsertTable *table, int *offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	if(index < size){
		int offset = offsets[index];
		pq->writeToNode(table->elements[offset],table->num_elements[offset],table->target[offset]);
	}
}
