#pragma once
#include <glad/glad.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include <vector>

#include "Shader.h"

class OctreeNode {
public:

	// Pointers the this nodes children
	OctreeNode* children[8];

	// Index of this node used when flattening the tree
	unsigned int index;
	unsigned int childIndices[8];

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

	// Mass and center of mass
	float m;
	float com_x;
	float com_y;
	float com_z;

	OctreeNode(){ ; };

	/**
	* OctreeNode constructor
	* @param min_x		min x bound
	* @param min_y		min y bound
	* @param min_z		min z bound
	* @param max_x		min x bound
	* @param max_y		min y bound
	* @param max_z		min z bound
	*/
	OctreeNode(float min_x, float min_y, float min_z,
		float max_x, float max_y, float max_z);

	~OctreeNode();

	/**
	* Returns the number of elements contained in this node
	* @return no_elements
	*/
	int getNoElements(){ return no_elements; }

	/**
	* Gets the child at position 'index'
	* @return child of index 'index'
	*/
	OctreeNode *getChild(int index){ return children[index]; }

	/**
	* Recursively steps through the tree and sets model matrix for each node
	* @param boxShader	shader instance in which the particle should be rendered
	*/
	void setModelAndRender(Shader boxShader);

	/**
	* Inserts a node into this octree
	* @param x		x-pos of node
	* @param y		y-pos of node
	* @param z		z-pos of node
	* @param mass	mass of node
	* @param com_x	center of mass in x-axis
	* @param com_y	center of mass in x-axis
	* @param com_z	center of mass in x-axis
	*/
	int insert(float x, float y, float z, float mass, float com_x, float com_y, float com_z);

	/**
	* Recursively steps through the tree and deallocates it
	*/
	void freeTree();


private:

	// Array storing pointers to children
	GLuint BoxVBO, BoxVAO, BoxEBO;

	// Position of this node if node is a leaf (particle)
	float pos_y;
	float pos_x;
	float pos_z;
	

	// Number of leaves contained in this node
	int no_elements;


	// Helper function to insert the node in the correct position
	int insert_sub(float x, float y, float z, float pmass, float pcom_x, float pcom_y, float pcom_z);
};