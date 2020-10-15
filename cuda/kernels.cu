#include "kernels.cuh"

__global__ void find_maximum_kernel(float *array, float *max, int *mutex, int N) {
	int start_i = threadIdx.x + blockIdx.x*blockDim.x;
	int step = gridDim.x*blockDim.x;

	__shared__ float cache[256];

	float temp_max = -1.0f;
	for(int i = start_i; i < N; i += step){
		temp_max = fmaxf(temp_max, array[i]);
	}

	cache[threadIdx.x] = temp_max;

	__syncthreads();

	// reduction
	for(int i = blockDim.x/2 ; i != 0 ; i /= 2){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

