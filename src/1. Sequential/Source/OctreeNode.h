#pragma once
#include <glad/glad.h>
#include <stdio.h>

class OctreeNode {
private:
	
	// Array storing pointers to children
	OctreeNode* children[8];

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



	int insert_sub(OctreeNode *node, float x, float y, float z, void *usr_data);

public:
	
	OctreeNode(float min_x, float min_y, float min_z,
				float max_x, float max_y, float max_z);

	~OctreeNode();

	// Inserts a node
	int insert(OctreeNode *node, float x, float y, float z, void *usr_val);

	void renderBounds();
};