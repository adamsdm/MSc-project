#pragma once

# define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <glad/glad.h>

#include "Particle.h"
#include "OctreeNode.h"

class OpenCLSim {
public:
	OpenCLSim();
	~OpenCLSim();

	void step();

	void updPos(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt);
	void updFor(Particle *ParticlesContainer, OctreeNode nodeContainer[], int count, int MAX_PARTICLES, float dt);

private:
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Kernel createKernel(char* filepath, char* name);
	cl::Context *context;

	cl::Kernel updPosKernel;
	cl::Kernel updForceKernel;
	cl::Kernel vectorAddKernel;

};