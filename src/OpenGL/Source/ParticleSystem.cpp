#include "ParticleSystem.h"


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
		p.pos = glm::vec3(x, y, z);

		if (i % 2 == 0){
			p.pos.x += 200;
			p.speed = 4.0f * glm::cross(-p.pos, glm::vec3(0.0, 0.0, 1.0));
		}
		else {
			p.pos.x -= 200;
			p.speed = 4.0f * glm::cross(-p.pos, glm::vec3(0.0, 0.0, 1.0));
		}

		
			

		ParticlesContainer[i] = p;

		// Setup position in buffer
		g_particule_position_size_data[i * 3 + 0] = p.pos.x;
		g_particule_position_size_data[i * 3 + 1] = p.pos.y;
		g_particule_position_size_data[i * 3 + 2] = p.pos.z;
	}
}


ParticleSystem::~ParticleSystem(){
	delete[] g_particule_position_size_data;
	delete[] ParticlesContainer;
}


// These functions should launch the kernels for the respective framework
void ParticleSystem::updateForces(float dt){



	for (int i = 0; i < MAX_PARTICLES; i++) {
		
		Particle pi = ParticlesContainer[i];
		float Fx = 0; float Fy = 0; float Fz = 0;

		for (int j = 0; j < MAX_PARTICLES; j++){

			if (i != j){
				Particle pj = ParticlesContainer[j];

				float dx = pj.pos.x - pi.pos.x;
				float dy = pj.pos.y - pi.pos.y;
				float dz = pj.pos.z - pi.pos.z;

				float dist = sqrt(dx*dx + dy*dy + dz*dz);
				
				// Fij = (G*mi*mj * (pj.pos - pi.pos)) / (||)
				float F = (9.82 * pi.weight * pj.weight) / (dist + SOFTENING * SOFTENING);

				Fx += F * dx / dist;
				Fy += F * dy / dist;
				Fz += F * dz / dist;
			}
		}

	
		// Update speed
		pi.speed.x += Fx;
		pi.speed.y += Fy;
		pi.speed.z += Fz;

		ParticlesContainer[i] = pi;
	}

}

void ParticleSystem::updatePositions(float dt){

	float simspeed = 0.01f;
		
	for (int i = 0; i < MAX_PARTICLES; i++){
		Particle p = ParticlesContainer[i];

		p.pos = glm::vec3(p.pos.x + p.speed.x * simspeed*dt, p.pos.y + p.speed.y * simspeed*dt, p.pos.z + p.speed.z*simspeed*dt);

		ParticlesContainer[i] = p;

		// Update position buffer
		g_particule_position_size_data[i * 3 + 0] = p.pos.x;
		g_particule_position_size_data[i * 3 + 1] = p.pos.y;
		g_particule_position_size_data[i * 3 + 2] = p.pos.z;
	}
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

	//glDrawArrays(GL_TRIANGLES, 0, 36);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, MAX_PARTICLES);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}
