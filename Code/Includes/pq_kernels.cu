__global__ void print_val(PriorityQueue *pq)
{
	printf("%d %d\n",threadIdx.x,pq->getSize());
}

__global__ void insert(PriorityQueue *pq, InsertTable *table,int* offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	if(index < size){
		int offset = offsets[index];
		if(table->indices[offset] == table->target[offset]){
			
			pq->writeToNode(table->elements[offset],table->num_elements[offset],table->target[offset]); // Make this atomic
			table->status[offset] = 0;
			table->num_elements[offset] = 0;
			
		}
		else{
			Node arr[2*NUM_PER_NODE];
			int tot_size = 0;
			Node tmp;
			for(int i=0;i<table->num_elements[offset];i++){
				arr[tot_size++] = table->elements[offset][i];
			}
			int node_num = table->indices[offset];
			for(int i=0;i<pq->nodes[node_num].size;i++){
				arr[tot_size++] = pq->nodes[node_num].nodes[i];
			}
			for(int i=0;i<tot_size;i++){
				for(int j=i+1;j<tot_size;j++){
					if(arr[i].key > arr[j].key){
						tmp = arr[i];
						arr[i] = arr[j];
						arr[j] = tmp;
					}
				}
			}
			for(int i=0;i<NUM_PER_NODE&&i<tot_size;i++){
				pq->nodes[node_num].nodes[i] = arr[i];
			}
			if(tot_size>NUM_PER_NODE){
				for(int i=NUM_PER_NODE;i<tot_size;i++){
					table->elements[offset][i-NUM_PER_NODE] = arr[i];
				}
			}
			// Change the entry so that it moves towards the target.
			int level = table->level[offset];
			//int target = table->target[offset];
			int rem;
			rem = table->target_bits[offset][level];
			table->indices[offset] = (node_num)*2+1+rem;
			table->level[offset] += 1;
			table->num_elements[offset] = (tot_size>NUM_PER_NODE)?tot_size-NUM_PER_NODE:0;
		}
	}
}
/*
__global__ void delete_update(PriorityQueue *pq, DeleteTable *table, int* offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	if(index < size){
		int offset = offsets[index];
		int num_node = table->indices[offset];
		int left = 2*num_node + 1;
		int right = 2*num_node + 2;
		int arr[3*NUM_PER_NODE];
		int tot_size = 0;
		int done = 0;
		int tmp;
		if(pq->nodes[left].size == 0 && pq->nodes[right].size == 0){
			table->status[offset] = 0;
			done = 1;
		}
		else{
			for(int i=0;i<pq->nodes[num_node].size;i++){
				arr[tot_size++] = pq->nodes[num_node].nodes[i].key;
			}
			for(int i=0;i<pq->nodes[left].size;i++){
				arr[tot_size++] = pq->nodes[left].nodes[i].key;
			}
			for(int i=0;i<pq->nodes[right].size;i++){
				arr[tot_size++] = pq->nodes[right].nodes[i].key;
			}
			if(pq->nodes[right].size==0 && pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key < pq->nodes[left].nodes[0].key){
					table->status[offset] = 0;
					done = 1;
			}
			else if(pq->nodes[left].size==0 && pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key < pq->nodes[right].nodes[0].key){
				table->status[offset] = 0;
				done = 1;
			}
			else if(pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key < pq->nodes[right].nodes[0].key && pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key < pq->nodes[left].nodes[0].key){
				table->status[offset] = 0;
				done = 1;
			}
			
			if(done != 1){
				for(int i=0;i<tot_size;i++){
					for(int j=i+1;j<tot_size;j++){
						if(arr[i] > arr[j]){
							tmp = arr[i];
							arr[i] = arr[j];
							arr[j] = tmp;
						}
					}
				}
				for(int i=0;i<NUM_PER_NODE&&i<tot_size;i++){
					pq->nodes[num_node].nodes[i].key = arr[i];
				}
				
				if(pq->nodes[right].nodes[pq->nodes[right].size-1].key > pq->nodes[left].nodes[pq->nodes[left].size-1].key){
					tmp = left;
					left = right;
					right = tmp;
				}
				
				pq->deleteUpdate(arr+NUM_PER_NODE,NUM_PER_NODE,left);
				pq->deleteUpdate(arr+2*NUM_PER_NODE,NUM_PER_NODE,right);
				pq->nodes[num_node].size = (tot_size < NUM_PER_NODE)?tot_size:NUM_PER_NODE;
				pq->nodes[left].size = (tot_size < 2*NUM_PER_NODE)?tot_size-NUM_PER_NODE:NUM_PER_NODE;
				pq->nodes[right].size = (tot_size < 3*NUM_PER_NODE)?tot_size-2*NUM_PER_NODE:NUM_PER_NODE;
				table->indices[offset] = right;
				table->level[offset] += 1;
				
				
			}
		}
	}
}*/

/*__global__ void writeToNode(PriorityQueue* pq, InsertTable *table, int *offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	if(index < size){
		int offset = offsets[index];
		pq->writeToNode(table->elements[offset],table->num_elements[offset],table->target[offset]);
	}
}*/
