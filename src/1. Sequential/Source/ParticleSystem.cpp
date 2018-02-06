﻿#include "ParticleSystem.h"


ParticleSystem::ParticleSystem(static const unsigned int _MAX_PARTICLES) {
	MAX_PARTICLES = _MAX_PARTICLES;
	// Quad vertices
	GLfloat g_vertex_buffer_data[] = {
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, 0.5f, 0.0f,
		0.5f, 0.5f, 0.0f,
	};

	g_particule_position_size_data = new GLfloat[MAX_PARTICLES * 4];
	
	ParticlesContainer = new Particle[_MAX_PARTICLES];
	initParticleSystem();

	
	
	
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// The VBO containing the quad/cube
	glGenBuffers(1, &billboard_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	// The VBO containing the positions and sizes of the particles
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 4 * sizeof(GLfloat), g_particule_position_size_data, GL_STREAM_DRAW);
	
}

void ParticleSystem::initParticleSystem(){

	float phi, r, x, y, z;

	// Populate initial positions
	for (int i = 0; i < MAX_PARTICLES; i++) {

		Particle p = ParticlesContainer[i];
		

		phi = (float) rand() / RAND_MAX * 2.0f * M_PI;
		r = (float) rand() / RAND_MAX * MAX_DISTANCE;


		float x = r * cos(phi);
		float y = 10.0f + r * sin(phi);
		float z =  (rand() % (2 * 40) - (float)40);


		// Setup particle
		p.weight = 1.0f;
		p.px = x;
		p.py = y;
		p.pz = z;

		if (i % 2 == 0){
			p.px += 200;
		}
		else {
			p.px -= 200;
		}

		glm::vec3 speed = 4.0f * glm::cross(-glm::vec3(p.px, p.py, p.pz), glm::vec3(0.0, 0.0, 1.0));
		
		p.vx = speed.x;
		p.vy = speed.y;
		p.vz = speed.z;
			

		ParticlesContainer[i] = p;

		// Setup position in buffer
		g_particule_position_size_data[i * 3 + 0] = p.px;
		g_particule_position_size_data[i * 3 + 1] = p.py;
		g_particule_position_size_data[i * 3 + 2] = p.pz;
	}
}

ParticleSystem::~ParticleSystem(){
	delete[] g_particule_position_size_data;
	delete[] ParticlesContainer;
}

void ParticleSystem::getBounds(float &_minx, float &_maxx, float &_miny, float &_maxy, float &_minz, float &_maxz){


	float minx = 99999999.0f;
	float maxx = -99999999.0f;
	
	float miny = 99999999.0f;
	float maxy = -99999999.0f;

	float minz = 99999999.0f;
	float maxz = -99999999.0f;

	for (int i = 0; i < MAX_PARTICLES; i++){
		glm::vec3 pos = glm::vec3(ParticlesContainer[i].px, ParticlesContainer[i].py, ParticlesContainer[i].pz);
		
		minx = std::min(pos.x, minx);
		maxx = std::max(pos.x, maxx);

		miny = std::min(pos.y, miny);
		maxy = std::max(pos.y, maxy);

		minz = std::min(pos.z, minz);
		maxz = std::max(pos.z, maxz);
	}


	_minx = minx;
	_maxx = maxx;

	_miny = miny;
	_maxy = maxy;

	_minz = minz;
	_maxz = maxz;
	
}

void ParticleSystem::renderBounds(){
	float minx, maxx;
	float miny, maxy;
	float minz, maxz;

	getBounds(minx, maxx, miny, maxy, minz, maxz);

	// Bad way.. :/
	float vertices[] = {
		minx, miny, minz,
		maxx, miny, minz,
		maxx, maxy, minz,
		maxx, maxy, minz,
		minx, maxy, minz,
		minx, miny, minz,

		minx, miny, maxz,
		maxx, miny, maxz,
		maxx, maxy, maxz,
		maxx, maxy, maxz,
		minx, maxy, maxz,
		minx, miny, maxz,

		maxx, maxy, maxz,
		minx, maxy, minz,
		minx, miny, minz,
		minx, miny, minz,
		minx, miny, maxz,
		minx, maxy, maxz, 

		maxx, maxy, maxz, 
		maxx, maxy, minz,
		maxx, miny, minz,
		maxx, miny, minz,
		maxx, miny, maxz,
		maxx, maxy, maxz,

		minx, miny, minz,
		maxx, miny, minz,
		maxx, miny, maxz,
		maxx, miny, maxz,
		minx, miny, maxz,
		minx, miny, minz,

		minx, maxy, minz,
		maxx, maxy, minz,
		maxx, maxy, maxz,
		maxx, maxy, maxz,
		maxx, maxy, maxz,
		minx, maxy, minz,
	};
	

	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// position attribute

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawArrays(GL_TRIANGLES, 0, 36);
}

void ParticleSystem::render(float dt){
	
	updateForces(dt);
	updatePositions(dt);

		
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, MAX_PARTICLES * sizeof(GLfloat) * 4, g_particule_position_size_data);

	
	// Setup attributes for the particle shader
	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	

	// 2nd attribute buffer : positions of particles' centers
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
	glVertexAttribDivisor(1, 1); // positions : one per quad (its center)                 -> 1

	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, MAX_PARTICLES);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}



// These functions should launch the kernels for the respective framework
void ParticleSystem::updateForces(float dt){



	for (int i = 0; i < MAX_PARTICLES; i++) {

		Particle pi = ParticlesContainer[i];
		float Fx = 0; float Fy = 0; float Fz = 0;

		for (int j = 0; j < MAX_PARTICLES; j++){

			if (i != j){
				Particle pj = ParticlesContainer[j];

				float dx = pj.px - pi.px;
				float dy = pj.py - pi.py;
				float dz = pj.pz - pi.pz;

				float dist = sqrt(dx*dx + dy*dy + dz*dz);

				float F = (9.82 * pi.weight * pj.weight) / (dist + SOFTENING * SOFTENING);

				Fx += F * dx / dist;
				Fy += F * dy / dist;
				Fz += F * dz / dist;
			}
		}


		// Update speed
		pi.vx += Fx;
		pi.vy += Fy;
		pi.vz += Fz;

		ParticlesContainer[i] = pi;
	}

}

void ParticleSystem::updatePositions(float dt){

	float simspeed = 0.01f;


	for (int i = 0; i < MAX_PARTICLES; i++){
		Particle p = ParticlesContainer[i];

		p.px = p.px + p.vx * simspeed*dt;
		p.py = p.py + p.vy * simspeed*dt;
		p.pz = p.pz + p.vz * simspeed*dt;


		ParticlesContainer[i] = p;

		// Update position buffer
		g_particule_position_size_data[i * 3 + 0] = p.px;
		g_particule_position_size_data[i * 3 + 1] = p.py;
		g_particule_position_size_data[i * 3 + 2] = p.pz;
	}
}
