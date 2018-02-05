#pragma once
#include <stdio.h>
#include <glad/glad.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <amp_math.h>
#include <glm/glm.hpp>

#define M_PI	3.14159265359
#define SOFTENING 1e-9f


struct Particle{
	glm::vec3 pos, speed;
	float weight;
};

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

	void initParticleSystem();
	void updateForces(float dt);
	void updatePositions(float dt);

public:

	ParticleSystem(const unsigned int _MAX_PARTICLES = 1000);
	~ParticleSystem();
	
	void render(float dt);

};