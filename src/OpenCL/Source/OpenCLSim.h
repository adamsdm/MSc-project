#pragma once

# define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <glad/glad.h>

#include "Particle.h"
#include "OctreeNode.h"
#include "sOctreeNode.h"
#include "config.h"

class OpenCLSim {
public:
	OpenCLSim();
	~OpenCLSim();

	/**
	* Launches a kernel that updates the positions of the particles.
	* @param *ParticlesContainer	array containing the particles
	* @param *g_particule_position_size_data	OpenGL position buffer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param dt	the stepsize, delta time
	*/
	void updPos(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt);

	/**
	* Launches a kernel that updates the forces of the particles.
	* @param *ParticlesContainer	array containing the particles
	* @param nodeContainer	array containing the nodes of the octree
	* @param count	the number of nodes in the nodeContainer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param dt	the stepsize, delta time
	*/
	void updFor(Particle *ParticlesContainer, sOctreeNode *nodeContainer, int count, int MAX_PARTICLES, float dt);

	/**
	* Calculates one simulation step without copying data back and forth more than necessary.
	* @param *p_container	array containing the particles
	* @param nodeContainer	array containing the nodes of the octree
	* @param *g_particule_position_size_data	OpenGL position buffer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param count	the number of nodes in the nodeContainer
	* @param dt	the stepsize, delta time
	*/
	void step(Particle *ParticlesContainer, sOctreeNode *nodeContainer, GLfloat *g_particule_position_size_data, int count, unsigned int MAX_PARTICLES, float dt);

private:

	// Vector contaning available platforms on the system
	std::vector<cl::Platform> platforms;

	// Vector containing available devices on the system
	std::vector<cl::Device> devices;

	// Default platform (platforms[0])
	cl::Platform default_platform;

	// Default device (devices[0])
	cl::Device default_device;

	/**
	* Reads and compiles a kernel from source 
	* @param *filepath	path to the kernel
	* @param name	name of the kernel function in the kernel code
	*/
	cl::Kernel createKernel(char* filepath, char* name);
	
	// OpenCL context, created in constructor from default device
	cl::Context *context;

	// Kernels
	cl::Kernel updPosKernel;
	cl::Kernel updForceKernel;
	cl::Kernel vectorAddKernel;

};