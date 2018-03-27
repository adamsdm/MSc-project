#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

const unsigned int SIZE = 1024;

// addition kernel
__global__ void add(const int *in_a, const int *in_b, int *out)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < SIZE)
		out[idx] = in_a[idx] + in_b[idx];
}

int main(){

	// Host pointers for io data
	int *a = new int[SIZE];
	int *b = new int[SIZE];
	int *c = new int[SIZE];

	// Device pointers
	int *d_a, *d_b, *d_c;

	for (int i = 0; i < SIZE; i++){
		a[i] = i;
		b[i] = 2*i;
	}

	// Allocate memory on the device
	const unsigned int size = SIZE * sizeof(int);
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);
	
	// Copy the input data to the device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	dim3 dimGrid(1);
	dim3 dimBlock(SIZE);

	add <<<dimGrid, dimBlock >>> (d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < SIZE; i++){
		if (a[i] + b[i] != c[i])
			return 1;
	}
	std::cout << "Sucess!" << std::endl;

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete[] a; delete[] b; delete[] c;

	return 0;
}