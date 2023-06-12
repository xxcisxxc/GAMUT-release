#include "heap.h"

__device__
void example6::min_heapify(volatile float *arr, int ind, int heap_size)
{
	int left = 2 * ind + 1;
	int right = 2 * ind + 2;
	int smallest = 0;
	if (left < heap_size && arr[left] < arr[ind])
		smallest = left;
	else
		smallest = ind;
	if (right < heap_size && arr[right] < arr[smallest])
		smallest = right;
	if (smallest != ind) {
		float temp = arr[ind];
		arr[ind] = arr[smallest];
		arr[smallest] = temp;
		min_heapify(arr, smallest, heap_size);
	}
}

__device__
void example6::build_min_heap(volatile float *arr, int heap_size)
{
	for (int i = heap_size / 2 - 1; i >= 0; i--)
		min_heapify(arr, i, heap_size);
}

__device__
void example6::min_heap_push_pop(volatile float *arr, float val, int heap_size)
{
	arr[0] = val;
	min_heapify(arr, 0, heap_size);
}