#define MAX_DEPTH 4
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
					if(arr[i].key <= arr[j].key){
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
			int rem;
			rem = table->target_bits[offset][level];
			table->indices[offset] = (node_num)*2+1+rem;
			table->level[offset] += 1;
			table->num_elements[offset] = (tot_size>NUM_PER_NODE)?tot_size-NUM_PER_NODE:0;
		}
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
			if(pq->nodes[right].size==0 && pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key > pq->nodes[left].nodes[0].key){
					table->status[offset] = 0;
					done = 1;
			}
			else if(pq->nodes[left].size==0 && pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key > pq->nodes[right].nodes[0].key){
				table->status[offset] = 0;
				done = 1;
			}
			else if(pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key > pq->nodes[right].nodes[0].key && pq->nodes[num_node].nodes[pq->nodes[num_node].size-1].key > pq->nodes[left].nodes[0].key){
				table->status[offset] = 0;
				done = 1;
			}
			if(done != 1){
				for(int i=0;i<tot_size;i++){
					for(int j=i+1;j<tot_size;j++){
						if(arr[i].key <= arr[j].key){
							tmp_node = arr[i];
							arr[i] = arr[j];
							arr[j] = tmp_node;
						}
					}
				}
				for(int i=0;i<NUM_PER_NODE&&i<tot_size;i++){
					pq->nodes[num_node].nodes[i] = arr[i];
				}
				
				if(pq->nodes[right].nodes[pq->nodes[right].size-1].key < pq->nodes[left].nodes[pq->nodes[left].size-1].key){
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
	Connect4State *c4s = new Connect4State();
	for(int i=0;i<size;i++){
		c4s = c4s->makeMove(moves[i]);
	}
	Node root((INT_MAX-1)/2,c4s);
	root.val->setRoot(true);
	root.val->setDepth(0);
	d_to_insert[0] = root;
}

__global__ void sss_star_algo(Node *d_to_send,int num_to_send,Node *d_to_insert, int *num_inserts, bool *isEnd, int *bestMove, bool player){
	int index = threadIdx.x +blockIdx.x*blockDim.x;
	int location;
	if(index < num_to_send){
		Node node = d_to_send[index];
		Connect4State *state;
		state = node.val;
		//state->printState();
		//printf("%d\n",state->heuristicEval(player));
		if(state->getDepth() == 0 && state->getSolved()){
			*isEnd = true;
			*bestMove = state->bestMove;
		}		
		 else if(!state->getSolved()){
			if(!state->getOver()&&state->getDepth()<MAX_DEPTH){
				if((state->getTurn() == player)){
					// MAX node
					int num_moves = 0;
					state->moveGen(&num_moves);
					
					int moves_done = 0;
					for(int i=0;i<BOARD_SIZE;i++){
						if(state->moves[i]==1){
							location = atomicAdd(num_inserts,1);
							if(node.key > 100){
								node.key--;
							}
							Node tmp_node(node.key,state->makeMove(i));
							tmp_node.val->setRoot(false);
							d_to_insert[location] = tmp_node;
							moves_done++;
						}
					}
				}
				else{
					int num_moves = 1;
					
					if(!(state->parent_node->getSolved())){
						state->moveGen(&num_moves);
						for(int i=0;i<BOARD_SIZE;i++){
							if(state->moves[i]==1){
								location = atomicAdd(num_inserts,1);
								if(node.key > 100){
									node.key--;
								}
								Node tmp_node(node.key,state->makeMove(i));
								tmp_node.val->setRoot(false);
								d_to_insert[location] = tmp_node;
								break;
							}
						}
					}
				}
			}
			else{
				int num_moves = 1;
				location = atomicAdd(num_inserts,num_moves);
				int val = node.val->heuristicEval(player);
				node.key = (node.key<val)?node.key:val;
				node.val->setSolved(true);
				d_to_insert[location] = node;
			}
		}
		else{
			if((state->getTurn()==player)){
				if(node.val->isLastChild()){
					int num_moves = 1;
					location = atomicAdd(num_inserts,num_moves);
					Connect4State *parent;
					parent = node.val->parent_node;
					parent->setSolved(true);
					Node tmp_node(node.key,parent);
					d_to_insert[location] = tmp_node;
				}
				else{
					int num_moves = 1;
					location = atomicAdd(num_inserts,num_moves);
					Connect4State *parent,*next_child;
					parent = node.val->parent_node;
					int move = node.val->getNextChild();
					next_child = parent->makeMove(move);
					next_child->setSolved(false);
					Node tmp_node(node.key,next_child);
					d_to_insert[location] = tmp_node;
				}
			}
			else{
					int num_moves = 1;
					location = atomicAdd(num_inserts,num_moves);
					Connect4State *parent;
					parent = node.val->parent_node;
					parent->setSolved(true);
					Node tmp_node(node.key,parent);
					d_to_insert[location] = tmp_node;
					parent->bestMove = node.val->child_num;
					
			}
			
		}
	}
}
