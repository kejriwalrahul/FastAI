#define R 3
#define QSIZE 1000
#ifndef _KERNELS_H_
#define _KERNELS_H_

class Node{
	int key;
	int val;

public:	
	__host__ __device__
	Node(int a,int b){
		key = a;
		val = b;
	}
	__host__ __device__
	Node(){
		key = 0;
		val = 0;
	}
	__host__ __device__
	int getKey(){
		return key;
	}
	
	__host__ __device__
	int getVal(){
		return val;
	}
	
};

class PQNode{
	Node nodes[R];
	int level;
	int size;
	
public:	
	__host__ __device__
	PQNode(Node *a, int len, int c){
		int min = (len < R)?len:R;
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
	PQNode nodes[QSIZE];
	int curr_size;
	
public:
	__host__ __device__
	PriorityQueue(){
		curr_size = 0;
	}
	
	__host__ __device__
	int getSize(){
		return curr_size;
	}
};

#endif
