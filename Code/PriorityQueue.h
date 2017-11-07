#define R 3
#define QSIZE 1000
#define NUM_BITS 10
#include <iostream>
#include <stdio.h>
using namespace std;
#ifndef _KERNELS_H_
#define _KERNELS_H_

class Node{
public:
	int key;
	int val;

	
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
public:
	Node nodes[R];
	int level;
	int size;
	
	
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
public:	
	PQNode nodes[QSIZE];
	int sizes[QSIZE];
	int curr_size;


	__host__ __device__
	PriorityQueue(){
		curr_size = 0;
		for(int i=0;i<QSIZE;i++){
			sizes[i] = 0;
		}
	}
	
	__host__ __device__
	int getSize(){
		return curr_size;
	}
	
	__host__ __device__
	int getInsertTarget(){
		int ans = -1;
		for(int i=0;i<QSIZE;i++){
			if(sizes[i]!=R){
				ans = i;
				break;
			}
		}
		return ans;
	}
	
	__host__ __device__
	void writeToNode(int *arr, int size, int index){
		int orig_size = nodes[index].size;
		for(int i=0;i<size&&nodes[index].size<R;i++){
			nodes[index].nodes[nodes[index].size].key = arr[i];
			nodes[index].size++;
		}
		if(orig_size == 0){
			curr_size++;
		}
	}
	
	__host__
	void print_object() {
		for(int i=0;i<curr_size;i++){
			cout << "Node number " << i << endl;
			for(int j=0;j<nodes[i].size;j++){
				cout << nodes[i].nodes[j].key << " ";
			}
			cout << endl;
		}
		cout << "Printed object" << endl;
	}
	
};

class InsertTable{
public:
	int status[QSIZE]; // Says whether the entry in the table is in use or not.
	int indices[QSIZE]; // Index of the node in the priority queue on which the process should happen.
	int elements[QSIZE][R]; // Set of elements to be inserted at the node.
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
	void addEntry(int index,int *elmts,int size,int tgt){
		int off = 0;
		while(status[off]!=0){
			off++;
		}
		status[off] = 1;
		indices[off] = index;
		int min = (size < R)?size:R;
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
					printf("%d ",elements[i][j]);
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

#endif
