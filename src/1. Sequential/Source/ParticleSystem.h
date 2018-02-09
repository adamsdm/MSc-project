#pragma once
#include <stdio.h>
#include <glad/glad.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <amp_math.h>
#include <glm/glm.hpp>
#include <limits>

#include "Camera.h"
#include "Shader.h"

#define M_PI	3.14159265359
#define SOFTENING 1e-9f


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
	
	// Buffers
	GLuint VertexArrayID;
	GLuint billboard_vertex_buffer;
	GLuint particles_position_buffer;
	GLfloat* g_particule_position_size_data;
	static const GLfloat g_vertex_buffer_data[];

	Particle* ParticlesContainer;
	Shader *particleShader;

	/**
	* Inits the positions and velocities of the particles.
	*/
	void initParticleSystem();

	/**
	* Sequentially calculates the forces.
	* @param dt delta time.
	*/
	void updateForces(float dt);

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
	void renderBounds(Shader boxShader);
	void renderCube();

	/**
	* Sequentially calculates the bounds of the particles.
	*/
	void getBounds(float &minx, float &maxx, float &miny, float &maxy, float &minz, float &maxz);

	Particle getParticle(int index){ return ParticlesContainer[index]; }

};