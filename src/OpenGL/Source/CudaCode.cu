#include "CudaCode.cuh"



typedef struct
{
	float px, py, pz;
	float vx, vy, vz;
	float weight;
} Particle;


__global__ void testKernel(Particle *p){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	p[i].px = 1.1;
	p[i].py = 2.2;
	p[i].pz = 3.3;

}

void temp::CudaTestStruct(){

	int numPoints = 16,
		gpuBlockSize = 4,
		pointSize = sizeof(Particle),
		numBytes = numPoints * pointSize,
		gpuGridSize = numPoints / gpuBlockSize;

	// allocate memory
	Particle *particles, *d_particles;
	particles = (Particle*) malloc(numBytes);
	cudaMalloc((void**)&d_particles, numBytes);

	// launch kernel
	testKernel << <gpuGridSize, gpuBlockSize >> >(d_particles);

	// retrieve the results
	cudaMemcpy(particles, d_particles, numBytes, cudaMemcpyDeviceToHost);
	printf("testKernel results:\n");
	for (int i = 0; i < numPoints; ++i)
	{
		printf("point.a: %f, point.b: %f\n", particles[i].px, particles[i].py);
	}
}