#define NUM_PER_NODE 3
#define QSIZE 1000
#define NUM_BITS 10
#ifndef _KERNELS_H_
#define _KERNELS_H_

class Node{
public:
	int key;
	TicTacToeState *val;

	
	__host__ __device__
	Node(int a,TicTacToeState *b){
		key = a;
		val = b;
	}
	__host__ __device__
	Node(){
		key = INT_MAX;
		val = NULL;
	}
	__host__ __device__
	int getKey(){
		return key;
	}
	
	__host__ __device__
	TicTacToeState* getVal(){
		return val;
	}
	
	__host__ __device__
	bool operator <(const Node& n) {
         if(key < n.key) {
            return true;
         }
         return false;
      }
	
};

__host__ __device__ bool operator<(const Node &lhs, const Node &rhs) { return (lhs.key < rhs.key); };

class PQNode{
public:
	Node nodes[NUM_PER_NODE];
	int level;
	int size;
	
	
	__host__ __device__
	PQNode(Node *a, int len, int c){
		int min = (len < NUM_PER_NODE)?len:NUM_PER_NODE;
		for(int i=0;i<min;i++){
			nodes[i] = a[i];
		}
		level = c;
		size = min;
	}
	
	__host__ __device__
	PQNode(){
		level = 0;
		size = 0;
	}
};

class PriorityQueue{
public:	
	PQNode nodes[QSIZE];
	int sizes[QSIZE];
	int curr_size;


	__host__ __device__
	PriorityQueue(){
		curr_size = 0;
		for(int i=0;i<QSIZE;i++){
			nodes[i].size = 0;
		}
	}
	
	__host__ __device__
	int getSize(){
		return curr_size;
	}
	
	__host__ __device__
	int getInsertTarget(int size, bool *done, int *inserted){
		int ans = -1;
		for(int i=0;i<QSIZE;i++){
			if(nodes[i].size!=NUM_PER_NODE){
				ans = i;
				*inserted = size<(NUM_PER_NODE - nodes[i].size)?size:NUM_PER_NODE - nodes[i].size;
				if(*inserted < size){
					*done = false;
				}
				else{
					*done = true;
				}
				break;
			}
		}
		return ans;
	}
	
	__host__ __device__
	void writeToNode(Node *arr, int size, int index){
		for(int i=0;i<size&&nodes[index].size<NUM_PER_NODE;i++){
			nodes[index].nodes[nodes[index].size] = arr[i];
			nodes[index].size++;
		}
		
		
	}
	
	__host__ __device__
	PQNode readRoot(){
		return nodes[0];
	}
	
	__host__ __device__
	void deleteUpdate(Node *arr, int size, int index){
		nodes[index].size = 0;
		for(int i=0;i<size&&i<NUM_PER_NODE&&arr[i].key<INT_MAX;i++){
			nodes[index].nodes[nodes[index].size] = arr[i];
			nodes[index].size++;
		}
	}
	
	__host__
	void print_object() {
		for(int i=0;i<QSIZE;i++){
			if(nodes[i].size>0){
				printf("Node number %d\n",i);
				for(int j=0;j<nodes[i].size;j++){
					printf("%d ",nodes[i].nodes[j].key);
				}
				printf("\n");
			}
		}
		printf("Printed object\n");
	}
	
};

class InsertTable{
public:
	int status[QSIZE]; // Says whether the entry in the table is in use or not.
	int indices[QSIZE]; // Index of the node in the priority queue on which the process should happen.
	Node elements[QSIZE][NUM_PER_NODE]; // Set of elements to be inserted at the node.
	int num_elements[QSIZE];
	int level[QSIZE]; // Level number of the node on which the operation occurs.
	int target[QSIZE]; // Target node to insert the set of elements.
	int target_bits[QSIZE][NUM_BITS];
	

	__host__ __device__
	InsertTable(){
		for(int i=0;i<QSIZE;i++){
			status[i] = 0;
			num_elements[i] = 0; 
		}
	}
	
	__host__ __device__
	void addEntry(int index,Node *elmts,int size,int tgt){
		int off = 0;
		while(status[off]!=0){
			off++;
		}
		status[off] = 1;
		indices[off] = index;
		int min = (size < NUM_PER_NODE)?size:NUM_PER_NODE;
		for(int i=0;i<min;i++){
			elements[off][i] = elmts[i];
			num_elements[off]++;
		}
		target[off] = tgt;
		int val = 0,o1 = index;
		while(o1 > 0){
			val++;
			o1/=2;
		}
		level[off] = val;
		int num = 0,rem;
		tgt+=1;
		while(tgt > 0){
			rem = tgt%2;
			tgt/=2;
			num = num*2+rem;
		}
		int count = 0;
		while(num>0){
			num/=2;
			target_bits[off][count++] = num%2;
		}
	}
	
	__host__ __device__
	void printTable(){
		int count = 0;
		for(int i=0;i<QSIZE;i++){
			if(status[i] == 1){
				printf("Index: %d\n",indices[i]);
				printf("Elements: ");
				for(int j=0;j<num_elements[i];j++){
					printf("%d ",elements[i][j].key);
				}
				printf("\nLevel: %d\n",level[i]);
				printf("Target: %d\n",target[i]);
				printf("##########\n");
				count++;
			}
		}
		printf("Done XXXX %d\n\n",count);
	}
	

};

class DeleteTable{
public:
	int status[QSIZE];
	int indices[QSIZE];
	int level[QSIZE];
	
	__host__ __device__
	DeleteTable(){
		for(int i=0;i<QSIZE;i++){
			status[i] = 0;
		}
	}
	
	__host__ __device__
	void addEntry(){
		int off = 0;
		while(status[off]!=0){
			off++;
		}
		status[off] = 1;
		indices[off] = 0;
		level[off] = 0;
	}
	
	__host__ __device__
	void printTable(){
		int count = 0;
		for(int i=0;i<QSIZE;i++){
			if(status[i] == 1){
				printf("Index: %d\n",indices[i]);
			}
		}
		printf("Done XXXX %d\n\n",count);
	}
	
};

#endif
