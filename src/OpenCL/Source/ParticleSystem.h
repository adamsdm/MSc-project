#pragma once
#include <stdio.h>
#include <glad/glad.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <amp_math.h>
#include <glm/glm.hpp>
#include <limits>
#include <queue>

#include "Camera.h"
#include "Shader.h"
#include "OctreeNode.h"
#include "sOctreeNode.h"
#include "Particle.h"
#include "Cell.h"
#include "MyTimer.h"
#include "OpenCLSim.h"



#define M_PI	3.14159265359
#define G		6.672e-11F
#define SOFTENING 1e-9f

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

	// OpenCL wrapper class
	OpenCLSim clSim;

	// Box buffers
	GLuint BoxVBO, BoxVAO, BoxEBO;

	// COM buffer
	GLuint comVBO, comVAO;

	// Data
	Particle *ParticlesContainer;
	OctreeNode* nodeContainer;
	sOctreeNode *sNodeContainer;

	Shader *particleShader;
	OctreeNode *root;

	/**
	* Inits the positions and velocities of the particles.
	*/
	void initParticleSystem();

	
	void buildTree();
	void calcTreeCOM(OctreeNode *node);

	/**
	* Flattens the tree into a std::vector in a breadth first order
	* @param *node root of tree
	*/
	void flattenTree(OctreeNode *node, int &count);

	/**
	* Sequentially calculates the forces.
	* @param dt delta time.
	*/
	void updateForces(float dt);
	void BarnesHutUpdateForces(float dt);
	void calcParticleForce(Particle &p, OctreeNode *root, float dt);


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

	OctreeNode* getTree(){ return root; }
	/**
	* Renders bounding box
	* @param boxShader shader to be used when rendering bounds
	*/
	void renderBounds(Shader boxShader);

	/**
	* Renders COM
	* @param *node pointer to root of tree
	* @param comShader shader to be used when rendering bounds
	*/
	void renderCOM(OctreeNode *node, Shader comShader);
	void renderCube();

	/**
	* Sequentially calculates the bounds of the particles.
	*/
	void getBounds(float &minx, float &maxx, float &miny, float &maxy, float &minz, float &maxz);

	Particle getParticle(int index){ return ParticlesContainer[index]; }

};