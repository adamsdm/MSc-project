#include "Cudacode.cuh"

#ifndef M_PI
	#define M_PI	3.14159265359
#endif

#ifndef G
	#define G		6.672e-11F
#endif

#ifndef SOFTENING
	#define SOFTENING 1e-9f
#endif



CudaSim::CudaSim(){
	;
}

CudaSim::~CudaSim(){
	;
}

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

void CudaSim::CUDAUpdatePositions(Particle *p_container, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt){

	
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

	updatePositionKernel <<< dimGrid, dimBlock >>>(d_positions, d_ParticlesContainer, MAX_PARTICLES, dt, simspeed);
	cudaThreadSynchronize();

	// retrieve the results
	gpuErrchk(cudaMemcpy(p_container, d_ParticlesContainer, size, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(g_particule_position_size_data, d_positions, buffer_size, cudaMemcpyDeviceToHost));
	

	cudaFree(d_ParticlesContainer);
	cudaFree(d_positions);

}


__device__ void devRecCalcParticleForce(Particle *p, OctreeNode *node, OctreeNode *nodeContainer, float dt){

	if (!node) return;

	float dx = node->com_x - p->px;
	float dy = node->com_y - p->py;
	float dz = node->com_z - p->pz;
	
	float dist = sqrt(dx*dx + dy*dy + dz*dz);

	if (dist == 0) return;
	

	float width = ((node->max_x - node->min_x) +
		(node->max_y - node->min_y) +
		(node->max_z - node->min_z)) / 3;

	// The node is far away enough to be evaluated as a single node
	if (width / dist < 0.5){

		float F = (G * p->weight * node->m) / (dist + 1.0f + SOFTENING * SOFTENING);

		p->vx += F * dx / dist;
		p->vy += F * dy / dist;
		p->vz += F * dz / dist;
	}

	

	// The node is to close to be treated as a single particle and must be further traversed
	else {
		if (node->childIndices[0]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[0]], nodeContainer, dt);
		if (node->childIndices[1]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[1]], nodeContainer, dt);
		if (node->childIndices[2]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[2]], nodeContainer, dt);
		if (node->childIndices[3]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[3]], nodeContainer, dt);
		if (node->childIndices[4]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[4]], nodeContainer, dt);
		if (node->childIndices[5]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[5]], nodeContainer, dt);
		if (node->childIndices[6]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[6]], nodeContainer, dt);
		if (node->childIndices[7]) devRecCalcParticleForce(p, &nodeContainer[node->childIndices[7]], nodeContainer, dt);
	}
	
}


__device__ struct MyQueue {
	static const int MAX_SIZE = 10000;
	int f = -1, r = -1;
	int A[MAX_SIZE];	// TODO: More dynamic array size

	
	__device__ bool empty() {
		return (f == -1 && r == -1);
	}

	__device__ bool isFull() {
		return (r + 1) % MAX_SIZE == f ? true : false;
	}

	__device__ void push(int x)
	{
		if (isFull())
		{
			printf("Error: Queue is Full\n");
			return;
		}
		if (empty())
		{
			f = r = 0;
		}
		else
		{
			r = (r + 1) % MAX_SIZE;
		}
		A[r] = x;
	}

	__device__ void pop()
	{
		if (empty())
		{
			printf("Error: Queue is Empty\n");
			return;
		}
		else if (f == r)
		{
			r = f = -1;
		}
		else
		{
			f = (f + 1) % MAX_SIZE;
		}
	}

	__device__ int front()
	{
		if (f == -1)
		{
			printf("Error: cannot return front from empty queue\n");
			return -1;
		}
		return A[f];
	}

	__device__ void print()
	{
		// Finding number of elements in queue  
		int count = (r + MAX_SIZE - f) % MAX_SIZE + 1;
		printf("Queue       : ");
		for (int i = 0; i <count; i++)
		{
			int index = (f + i) % MAX_SIZE; // Index of element while travesing circularly from front
			printf("%d ", A[index]);
		}
		printf("\n\n");
	}
};

__device__  __inline__ void devIterativeCalcParticleForce(Particle *p, OctreeNode *nodeContainer, float dt) {

	MyQueue q;

	// start by pushing the root, nodeContainer[0].index = 0
	q.push(nodeContainer[0].index);

	while (!q.empty()) {

		OctreeNode *node = &nodeContainer[q.front()];
		q.pop();

		float dx = node->com_x - p->px;
		float dy = node->com_y - p->py;
		float dz = node->com_z - p->pz;

		float dist = sqrt(dx*dx + dy*dy + dz*dz);

		if (dist == 0) return;


		float width = ((node->max_x - node->min_x) +
			(node->max_y - node->min_y) +
			(node->max_z - node->min_z)) / 3;

		// The node is far away enough to be evaluated as a single node
		if (width / dist < 0.5) {

			float F = (G * p->weight * node->m) / (dist + 1.0f + SOFTENING * SOFTENING);

			p->vx += F * dx / dist;
			p->vy += F * dy / dist;
			p->vz += F * dz / dist;
		}



		// The node is to close to be treated as a single particle and must be further traversed
		else {
			if (node->childIndices[0]) q.push(node->childIndices[0]);
			if (node->childIndices[1]) q.push(node->childIndices[1]);
			if (node->childIndices[2]) q.push(node->childIndices[2]);
			if (node->childIndices[3]) q.push(node->childIndices[3]);
			if (node->childIndices[4]) q.push(node->childIndices[4]);
			if (node->childIndices[5]) q.push(node->childIndices[5]);
			if (node->childIndices[6]) q.push(node->childIndices[6]);
			if (node->childIndices[7]) q.push(node->childIndices[7]);
		}
	}
}

__global__ void updateForceKernel(Particle *ParticlesContainer, OctreeNode *nodeContainer, int MAX_PARTICLES, float dt){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	if (i < MAX_PARTICLES){
		Particle *p = &ParticlesContainer[i];
		devIterativeCalcParticleForce(p, nodeContainer, dt);
		//devRecCalcParticleForce(p, &nodeContainer[0], nodeContainer, dt);
	}
}

void CUDACalcForces(Particle *ParticlesContainer, 
					OctreeNode nodeContainer[], 
					int count, 
					int MAX_PARTICLES, 
					float dt)
{

	OctreeNode *d_node_container;
	Particle *d_particle_container;


	gpuErrchk(cudaMalloc((void**)&d_particle_container, MAX_PARTICLES * sizeof(Particle)));
	gpuErrchk(cudaMemcpy(d_particle_container, ParticlesContainer, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&d_node_container, count * sizeof(OctreeNode)));
	gpuErrchk(cudaMemcpy(d_node_container, nodeContainer, count*sizeof(OctreeNode), cudaMemcpyHostToDevice));

	dim3 dimGrid(MAX_PARTICLES / 1024);
	dim3 dimBlock(1024);


	updateForceKernel <<<dimGrid, dimBlock >>> (d_particle_container, d_node_container, MAX_PARTICLES, dt);
	cudaThreadSynchronize();


	gpuErrchk((cudaMemcpy(ParticlesContainer, d_particle_container, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost)));
	
	gpuErrchk(cudaFree(d_node_container));
	gpuErrchk(cudaFree(d_particle_container));
}

void CudaSim::CUDAStep(Particle *p_container, OctreeNode nodeContainer[], GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, int count, float dt) {


	Particle *d_particle_container;
	OctreeNode *d_node_container;
	GLfloat *d_positions;

	
	// Container containing particles
	gpuErrchk(cudaMalloc((void**)&d_particle_container, MAX_PARTICLES * sizeof(Particle)));
	gpuErrchk(cudaMemcpy(d_particle_container, p_container, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));

	// Flattened tree container
	gpuErrchk(cudaMalloc((void**)&d_node_container, count * sizeof(OctreeNode)));
	gpuErrchk(cudaMemcpy(d_node_container, nodeContainer, count * sizeof(OctreeNode), cudaMemcpyHostToDevice));

	// Vertex buffer
	gpuErrchk(cudaMalloc((void**)&d_positions, MAX_PARTICLES * 3 * sizeof(GLfloat)));
	gpuErrchk(cudaMemcpy(d_positions, g_particule_position_size_data, MAX_PARTICLES * 3 * sizeof(GLfloat), cudaMemcpyHostToDevice));

	// Launch kernel
	float simspeed = 0.01;
	dim3 dimGrid(MAX_PARTICLES / 1024);
	dim3 dimBlock(1024);

	updateForceKernel <<<dimGrid, dimBlock >>> (d_particle_container, d_node_container, MAX_PARTICLES, dt);
	cudaThreadSynchronize();
	updatePositionKernel <<< dimGrid, dimBlock >> >(d_positions, d_particle_container, MAX_PARTICLES, dt, simspeed);
	cudaThreadSynchronize();


	// Retrieve result
	gpuErrchk((cudaMemcpy(p_container, d_particle_container, MAX_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost)));
	gpuErrchk(cudaMemcpy(g_particule_position_size_data, d_positions, MAX_PARTICLES * 3 * sizeof(GLfloat), cudaMemcpyDeviceToHost));
}