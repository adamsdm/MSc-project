#include "CudaCode.cuh"


__global__ void add(int *out, const int *in_a, const int *in_b)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < SIZE)
		out[idx] = in_a[idx] + in_b[idx];
}

void temp::CudaHello(){
	int *a = new int[SIZE];
	int *b = new int[SIZE];
	int *c = new int[SIZE];
	int *d_a, *d_b, *d_c;

	// Initialize input vectors
	for (size_t i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = 2 * i;
		c[i] = -1;
	}

	// Allocate and copy memory on device
	size_t size = SIZE * sizeof(int*);
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch kernel
	dim3 dimGrid(1);
	dim3 dimBlock(SIZE);

	add << <dimGrid, dimBlock >> > (d_c, d_a, d_b);
	cudaThreadSynchronize();
	
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	
	for (size_t i = 0; i < 10; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	delete a; delete b; delete c;
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}