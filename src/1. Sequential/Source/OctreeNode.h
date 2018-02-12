#pragma once
#include <glad/glad.h>
#include <stdio.h>
#include <glm/glm.hpp>

#include "Shader.h"

class OctreeNode {
private:
	
	// Array storing pointers to children
	OctreeNode* children[8];
	GLuint BoxVBO, BoxVAO, BoxEBO;

	// Position of this node if node is a leaf (particle)
	float pos_y;
	float pos_x;
	float pos_z;

	// Bounds
	float min_x;
	float min_y;
	float min_z;

	float max_x;
	float max_y;
	float max_z;

	float mid_x;
	float mid_y;
	float mid_z;

	// Pointer to used defined data
	void *usr_val;

	// Number of leaves contained in this node
	int no_elements;


	// Helper function to insert the node in the correct position
	int insert_sub(float x, float y, float z, void *usr_data);


	
	

public:
	
	OctreeNode(float min_x, float min_y, float min_z,
				float max_x, float max_y, float max_z);

	~OctreeNode();

	// Recursively steps through the tree and sets model matrix for each node
	// Should only be used from ParticleSystem.cpp
	void setModelAndRender(Shader boxShader);

	// Inserts a node into this octree
	int insert(float x, float y, float z, void *usr_val);
	void free();
};