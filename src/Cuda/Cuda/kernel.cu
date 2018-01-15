
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#define SIZE	1024

__global__ void add(int *out, const int *in_a, const int *in_b)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < SIZE)
		out[idx] = in_a[idx] + in_b[idx];
}

int main()
{

	float time = 0;

	int *a = new int[SIZE];
	int *b = new int[SIZE];
	int *c = new int[SIZE];
	int *d_a, *d_b, *d_c;

	// Initialize input vectors
	for (size_t i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}

	// Allocate and copy memory on device
	size_t size = SIZE * sizeof(int*);
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


	// Timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Launch kernel
	dim3 dimGrid(1);
	dim3 dimBlock(SIZE);

	cudaEventRecord(start);
	add << <dimGrid, dimBlock >> > (d_c, d_a, d_b);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	// Copy back result
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);


	// Assert correct result
	for (size_t i = 0; i < SIZE; i++) {
		if (c[i] != a[i] + b[i]) {
			std::cout << "ERROR AT INDEX " << i << ": \t" << a[i] << '+' << b[i] << "!=" << c[i] << std::endl;
			delete a; delete b; delete c;
			cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
			return EXIT_FAILURE;
		}
	}

	std::cout << "Success!" << std::endl;
	std::cout << "Done in " << time << "ms" << std::endl;

	delete a; delete b; delete c;
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);


	return EXIT_SUCCESS;
}