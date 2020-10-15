#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>

__global__ void find_maximum_kernel(float *array, float *max, int *mutex, int N, float start_max) {
    int start_i = threadIdx.x + blockIdx.x*blockDim.x;
    int step = gridDim.x*blockDim.x;

    __shared__ float cache[256];

    float temp_max = start_max;
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

int main() {
    int N = 1000*1000*20;
    float *host_array;
    float *device_array;
    float *host_max;
    float *device_max;
    int *device_mutex;


    // allocate memory
    host_array = (float*)malloc(N*sizeof(float));
    host_max = (float*)malloc(sizeof(float));
    cudaMalloc((void**)&device_array, N*sizeof(float));
    cudaMalloc((void**)&device_max, sizeof(float));
    cudaMalloc((void**)&device_mutex, sizeof(int));
    cudaMemset(device_max, 0, sizeof(float));
    cudaMemset(device_mutex, 0, sizeof(float));


    // fill host array with data
    srand(10);
    for(int i=0;i<N;i++){
        host_array[i] = 10000*float(rand()) / RAND_MAX-5000;
    }

    // set up timing variables
    float gpu_elapsed_time;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);


    // copy from host to device
    cudaEventRecord(gpu_start, 0);
    cudaMemcpy(device_array, host_array, N*sizeof(float), cudaMemcpyHostToDevice);


    // call kernel
    dim3 gridSize = 256;
    dim3 blockSize = 256;
    find_maximum_kernel<<< gridSize, blockSize >>>(device_array, device_max, device_mutex, N, host_array[0]);


    // copy from device to host
    cudaMemcpy(host_max, device_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);


    //report results
    std::cout<<"Maximum number found on gpu was: "<<*host_max<<std::endl;
    std::cout<<"The gpu took: "<<gpu_elapsed_time<<" milli-seconds"<<std::endl;


    // run cpu version
    clock_t cpu_start = clock();
    *host_max = host_array[0];
    for(int i=0;i<N;i++){
        if(host_array[i] > *host_max){
            *host_max = host_array[i];
        }
    }
    clock_t cpu_stop = clock();
    clock_t cpu_elapsed_time = 1000*(cpu_stop - cpu_start)/CLOCKS_PER_SEC;

    std::cout<<"Maximum number found on cpu was: "<<*host_max<<std::endl;
    std::cout<<"The cpu took: "<<cpu_elapsed_time<<" milli-seconds"<<std::endl;


    // free memory
    free(host_array);
    free(host_max);
    cudaFree(device_array);
    cudaFree(device_max);
    cudaFree(device_mutex);
}


