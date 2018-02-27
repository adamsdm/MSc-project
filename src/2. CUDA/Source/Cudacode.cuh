#pragma once

#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include <stdio.h>

#include <glad/glad.h>
#include "Particle.h"
#include "Cell.h"
#include "OctreeNode.h"
#include "MyTimer.h"

class CudaSim {
public:
	CudaSim();
	~CudaSim();

	/**
	* Launches a kernel that updates the positions of the particles.
	* @param *ParticlesContainer	array containing the particles
	* @param *g_particule_position_size_data	OpenGL position buffer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param dt	the stepsize, delta time
	*/
	void CUDAUpdatePositions(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt);

	/**
	* Launches a kernel that updates the forces of the particles.
	* @param *ParticlesContainer	array containing the particles
	* @param nodeContainer	array containing the nodes of the octree
	* @param count	the number of nodes in the nodeContainer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param dt	the stepsize, delta time
	*/
	void CUDACalcForces(Particle *ParticlesContainer, OctreeNode nodeContainer[], int count, int MAX_PARTICLES, float dt);

	/**
	* Calculates one simulation step without copying data back and forth more than necessary.
	* @param *p_container	array containing the particles
	* @param nodeContainer	array containing the nodes of the octree
	* @param *g_particule_position_size_data	OpenGL position buffer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param count	the number of nodes in the nodeContainer
	* @param dt	the stepsize, delta time
	*/
	void CUDAStep(Particle *p_container, OctreeNode nodeContainer[], GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, int count, float dt);

private:
	/**
	* Recursively traverses the flattened tree and calculates forces on the particle p
	* @param *p	the particle which force is calculated
	* @param *node	the node in the tree which the particle is compared to
	* @param *nodeContainer	the array containing all nodes in the octree
	* @param dt	the stepsize, delta time
	*/
	__device__ void devRecCalcParticleForce(Particle *p, OctreeNode *node, OctreeNode *nodeContainer, float dt);

	/**
	* Iteratively traverses the flattened tree and calculates forces on the particle p
	* @param *p	the particle which force is calculated
	* @param *node	the node in the tree which the particle is compared to
	* @param *nodeContainer	the array containing all nodes in the octree
	* @param dt	the stepsize, delta time
	*/
	__device__ void devIterativeCalcParticleForce(Particle *p, OctreeNode *node, OctreeNode *nodeContainer, float dt);

};
