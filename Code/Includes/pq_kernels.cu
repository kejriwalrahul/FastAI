__global__ void print_val(PriorityQueue *pq)
{
	printf("%d %d\n",threadIdx.x,pq->getSize());
}

__global__ void insert(PriorityQueue *pq, InsertTable *table,int* offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	
	
	if(index < size){
		//printf("%d\n",size);
		//table->printTable();
		
		int offset = offsets[index];
		//printf("Start: %d %d\n",table->indices[offset],table->target[offset]);
		if(table->indices[offset] == table->target[offset]){
			//printf("Here\n");
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
		//printf("End: %d %d\n",table->indices[offset],table->target[offset]);
	}
}

__global__ void delete_update(PriorityQueue *pq, DeleteTable *table, int* offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	if(index < size){
		int offset = offsets[index];
		int num_node = table->indices[offset];
		int left = 2*num_node + 1;
		int right = 2*num_node + 2;
		Node arr[3*NUM_PER_NODE];
		int tot_size = 0;
		int done = 0;
		int tmp;
		Node tmp_node;
		if(pq->nodes[left].size == 0 && pq->nodes[right].size == 0){
			table->status[offset] = 0;
			done = 1;
		}
		else{
			for(int i=0;i<pq->nodes[num_node].size;i++){
				arr[tot_size++] = pq->nodes[num_node].nodes[i];
			}
			for(int i=0;i<pq->nodes[left].size;i++){
				arr[tot_size++] = pq->nodes[left].nodes[i];
			}
			for(int i=0;i<pq->nodes[right].size;i++){
				arr[tot_size++] = pq->nodes[right].nodes[i];
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
						if(arr[i].key > arr[j].key){
							tmp_node = arr[i];
							arr[i] = arr[j];
							arr[j] = tmp_node;
						}
					}
				}
				for(int i=0;i<NUM_PER_NODE&&i<tot_size;i++){
					pq->nodes[num_node].nodes[i] = arr[i];
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
}

__global__ void createRootNode(Node *d_to_insert, int *moves, int size){
	TicTacToeState *ttts = new TicTacToeState();
	for(int i=0;i<size;i++){
		ttts = ttts->makeMove(moves[i]);
	}
	Node root(INT_MAX-1,ttts);
	root.val->setRoot(true);
	d_to_insert[0] = root;
}

__global__ void sss_star_algo(Node *d_to_send,int num_to_send,Node *d_to_insert, int *num_inserts, bool *isEnd){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	int location;
	if(index < num_to_send){
		Node node = d_to_send[index];
		TicTacToeState *state;
		state = node.val;
		//printf("%d %d %d\n",state->getSolved(),state->getOver(),state->getTurn());
		//printf("%d %d\n",node.key,state->getSolved());
		//	node.val->printState();
		if(state->getRoot() && state->getSolved()){
			*isEnd = true;
			printf("Solved root! %d\n",node.key);
		}		
		else if(!state->getSolved()){
			if(!state->getOver()){
				if(!state->getTurn()){
					// MAX node
					printf("Unsolved MAX node\n");
					int num_moves = 0;
					state->moveGen(&num_moves);
					num_moves++;
					location = atomicAdd(num_inserts,num_moves);
					int moves_done = 0;
					for(int i=0;i<BOARD_SIZE;i++){
						if(state->moves[i]==1){
							Node tmp_node(node.key-1,state->makeMove(i));
							tmp_node.val->setRoot(false);
							d_to_insert[location+moves_done] = tmp_node;
							moves_done++;
						}
					}
					d_to_insert[location+moves_done] = node;
					//printf("%d\n",d_to_insert[location+num_moves-1].key);
				}
				else{
					printf("Unsolved MIN node\n");
					int num_moves = 2;
					location = atomicAdd(num_inserts,num_moves);
					
					for(int i=0;i<BOARD_SIZE;i++){
						if(state->moves[i]==1){
							Node tmp_node(node.key-1,state->makeMove(i));
							tmp_node.val->setRoot(false);
							d_to_insert[location] = tmp_node;
							break;
						}
					}
					d_to_insert[location+1] = node;
					
				}
			}
			else{
				int num_moves = 1;
				location = atomicAdd(num_inserts,num_moves);
				int val = node.val->heuristicEval();
				node.key = (node.key<val)?node.key:val;
				node.val->setSolved(true);
				//printf("Here %d %d\n",node.val->getSolved(),node.key);
				d_to_insert[location] = node;
			}
		}
		else{
			printf("Solved node!!!\n");
			if(!state->getTurn()){
				if(node.val->isLastChild()){
					printf("Solved max last\n");
					int num_moves = 1;
					location = atomicAdd(num_inserts,num_moves);
					TicTacToeState *parent;
					parent = node.val->parent_node;
					parent->setSolved(true);
					Node tmp_node(node.key,parent);
					d_to_insert[location] = tmp_node;
				}
				else{
					printf("Solved max not last\n");
					int num_moves = 1;
					location = atomicAdd(num_inserts,num_moves);
					TicTacToeState *parent,*next_child;
					parent = node.val->parent_node;
					int move = node.val->getNextChild();
					next_child = parent->makeMove(move);
					next_child->setSolved(false);
					Node tmp_node(node.key,next_child);
					d_to_insert[location] = tmp_node;
				}
			}
			else{
				printf("I am min\n");
				int num_moves = 1;
				location = atomicAdd(num_inserts,num_moves);
				TicTacToeState *parent;
				parent = node.val->parent_node;
				parent->setSolved(true);
				Node tmp_node(node.key,parent);
				d_to_insert[location] = tmp_node;
			}
			
		}
		/*for(int i=0;i<*num_inserts;i++){
			printf("%d\n",d_to_insert[i].key);
			d_to_insert[i].val->printState();
		}*/
	}
}

/*__global__ void writeToNode(PriorityQueue* pq, InsertTable *table, int *offsets, int size){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	if(index < size){
		int offset = offsets[index];
		pq->writeToNode(table->elements[offset],table->num_elements[offset],table->target[offset]);
	}
}*/
