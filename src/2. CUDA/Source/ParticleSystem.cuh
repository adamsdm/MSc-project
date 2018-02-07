#pragma once

#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include <stdio.h>
#include <glad/glad.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <glm/glm.hpp>
#include <limits>

#define M_PI	3.14159265359
#define SOFTENING 1e-9f

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

typedef struct
{
	float px, py, pz;
	float vx, vy, vz;
	float weight;
} Particle;


class ParticleSystem  {
private:
	unsigned int MAX_PARTICLES = 1000;
	unsigned int MAX_DISTANCE = 100;

	// Buffer indices
	GLuint VertexArrayID;
	GLuint billboard_vertex_buffer;
	GLuint particles_position_buffer;

	// Buffer data
	GLfloat* g_particule_position_size_data;
	static const GLfloat g_vertex_buffer_data[];

	Particle* ParticlesContainer;

	/**
	* Inits the positions and velocities of the particles.
	*/
	void initParticleSystem();

	/**
	* Sequentially calculates the forces.
	* @param dt delta time.
	*/
	void updateForces(float dt);

	void CUDAStep(float dt);

	/**
	* Sequentially updates the positions.
	* @param dt delta time.
	*/
	void updatePositions(float dt);

public:

	ParticleSystem(const unsigned int _MAX_PARTICLES = 1000);
	~ParticleSystem();

	/**
	* Renders the particles instanced
	* @param dt delta time.
	*/
	void render(float dt);

	/**
	* Renders bounding box
	*/
	void renderBounds();

	/**
	* Sequentially calculates the bounds of the particles.
	*/
	void ParticleSystem::getBounds(float &minx, float &maxx, float &miny, float &maxy, float &minz, float &maxz);

};