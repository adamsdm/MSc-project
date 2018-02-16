#include "Cudacode.cuh"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		//printf("CUDAGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		printf("CUDA::ERROR %s line %d: %s\n", file, line, cudaGetErrorString(code));
		if (abort) exit(code);
	}
}


__global__ void updatePositionKernel(GLfloat *g_particule_position_size_data, Particle *ParticlesContainer, int MAX_PARTICLES, float dt, float simspeed){

	int i = blockIdx.x * blockDim.x + threadIdx.x;


	if (i < MAX_PARTICLES){

		Particle *p = &ParticlesContainer[i];

		p->px += p->vx * simspeed*dt;
		p->py += p->vy * simspeed*dt;
		p->pz += p->vz * simspeed*dt;

		// Update position buffer
		g_particule_position_size_data[i * 3 + 0] = p->px;
		g_particule_position_size_data[i * 3 + 1] = p->py;
		g_particule_position_size_data[i * 3 + 2] = p->pz;
	}


}

void CUDAUpdatePositions(Particle *p_container, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt){

	
	int size = MAX_PARTICLES * sizeof(Particle);
	float buffer_size = MAX_PARTICLES * 3 * sizeof(GLfloat);
	float simspeed = 0.01f;	// No more than ~0.1 for a stable simulation
	
	// Allocate memory
	Particle *d_ParticlesContainer;
	GLfloat *d_positions;

	// Particle container
	gpuErrchk(cudaMalloc((void**)&d_ParticlesContainer, size));
	gpuErrchk(cudaMemcpy(d_ParticlesContainer, p_container, size, cudaMemcpyHostToDevice));

	
	
	// Vertex buffer
	gpuErrchk(cudaMalloc((void**)&d_positions, buffer_size));
	gpuErrchk(cudaMemcpy(d_positions, g_particule_position_size_data, buffer_size, cudaMemcpyHostToDevice));

	// launch kernel
	dim3 dimGrid(MAX_PARTICLES / 1024);
	dim3 dimBlock(1024);

	updatePositionKernel << < dimGrid, dimBlock >> >(d_positions, d_ParticlesContainer, MAX_PARTICLES, dt, simspeed);
	cudaThreadSynchronize();

	// retrieve the results
	gpuErrchk(cudaMemcpy(p_container, d_ParticlesContainer, size, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(g_particule_position_size_data, d_positions, buffer_size, cudaMemcpyDeviceToHost));
	

	cudaFree(d_ParticlesContainer);
	cudaFree(d_positions);

}



struct point
{
	float a;
	point *lChild = nullptr;
	point *rChild = nullptr;
};


__global__ void updateForceKernel(OctreeNode *nodeContainer){
	
}

__global__ void testKernel(point *p)
{
	//printf("%f\n", p[0].lChild->a);
}

void CUDACalcForces(OctreeNode nodeContainer[]){
	
	int SIZE = 3;

	point *pointArray = (point*)malloc(SIZE * sizeof(point));
	point *d_pointArray;

	
	pointArray[0].a = 0.0f;
	pointArray[1].a = 1.1f;
	pointArray[2].a = 2.2f;

	pointArray[0].lChild = &pointArray[1];
	pointArray[0].lChild = &pointArray[2];

	cudaMalloc((void**)&d_pointArray, SIZE * sizeof(point));
	cudaMemcpy(d_pointArray, pointArray, SIZE * sizeof(point), cudaMemcpyHostToDevice);

	
	

	// launch kernel
	testKernel << <1, 1 >> >(d_pointArray);


	// deallocate memory
	free(pointArray);
	cudaFree(d_pointArray);



	/*
	OctreeNode* d_nodeContainer;
	gpuErrchk(cudaMalloc((void**)&d_nodeContainer, 4 * 2048 * sizeof(OctreeNode)));
	gpuErrchk(cudaMemcpy(d_nodeContainer, nodeContainer, 4 * 2048 * sizeof(OctreeNode), cudaMemcpyHostToDevice));
	updateForceKernel <<<1, 1 >>>(d_nodeContainer);
	
	cudaFree(d_nodeContainer);
	*/


}