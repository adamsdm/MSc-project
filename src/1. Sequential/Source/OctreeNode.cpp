// Implementation of a Octree which can contain both cells and particles (leafs)
// https://github.com/goakley/tcnj-coctree

#include "OctreeNode.h"


OctreeNode::OctreeNode(float _min_x, float _min_y, float _min_z,
						 float _max_x, float _max_y, float _max_z){
	
	// Calculate bounds and insure that they are correct
	min_x = (_min_x < _max_x ? _min_x : _max_x);
	min_y = (_min_y < _max_y ? _min_y : _max_y);
	min_z = (_min_z < _max_z ? _min_z : _max_z);

	max_x = (_min_x < _max_x ? _max_x : _min_x);
	max_y = (_min_y < _max_y ? _max_y : _min_y);
	max_z = (_min_z < _max_z ? _max_z : _min_z);
	
	mid_x = (min_x + max_x) / 2;
	mid_y = (min_y + max_y) / 2;
	mid_z = (min_z + max_z) / 2;

	pos_x = 0;
	pos_y = 0;
	pos_z = 0;

	for (size_t i = 0; i < 8; i++)
	{
		children[i] = nullptr;
	}
	
	usr_val = nullptr;
	no_elements = 0;
	

}

OctreeNode::~OctreeNode(){
	
}

int OctreeNode::insert(OctreeNode *node, float x, float y, float z, void *usr_data){

	if (!node) return 0;

	if (node->no_elements == 0){
		node->pos_x = x;
		node->pos_y = y;
		node->pos_z = z;
		node->usr_val = usr_val;
	}

	// If node already contains data, make node into a cell
	if (node->no_elements == 1){

	}

	no_elements++;
	return no_elements;
}

int OctreeNode::insert_sub(OctreeNode *node, float x, float y, float z, void* usr_data){
	if (!node) return 0;
	int sub = 0;
	float min_x, min_y, min_z;
	float max_x, max_y, max_z;

	// Select which octant to insert the node
	/* Children 0,2,4,8 have positive x-coordinates */
	if (x > node->mid_x) {
		sub += 1;
		min_x = node->mid_x;
		max_x = node->max_x;
	}
	else {
		min_x = node->min_x;
		max_x = node->mid_x;
	}

	/* Children 0,1,3,4 have positive y-coordinates */
	if (y > node->mid_y) {
		sub += 2;
		min_y = node->mid_y;
		max_y = node->max_y;
	}
	else {
		min_y = node->min_y;
		max_y = node->mid_y;
	}
	
	/* Children 0,1,2,3 have positive z-coordinates */

	if (z > node->mid_z) {
		sub += 4;
		min_z = node->mid_z;
		max_z = node->max_z;
	}
	else {
		min_z = node->min_z;
		max_z = node->mid_z;
	}

	if (!node->children[sub])
		node->children[sub] = new OctreeNode(min_x, min_y, min_z, max_x, max_y, max_z);

	return insert((node->children)[sub], x, y, z, usr_val);

}

void OctreeNode::renderBounds(){
	
	for (int i = 0; i < 8; i++){
		if (children[i])
			children[i]->renderBounds();
	}

	/*
	float vertices[] = {
		1.0f, 1.0f, 1.0f,  // front top right		0
		1.0f, -1.0f, 1.0f,  // front bottom right	1
		-1.0f, -1.0f, 1.0f,  // front bottom left	2
		-1.0f, 1.0f, 1.0f,   // front top left		3

		1.0f, 1.0f, -1.0f,  // back top right		4
		1.0f, -1.0f, -1.0f,  // back bottom right	5
		-1.0f, -1.0f, -1.0f,  // back bottom left	6
		-1.0f, 1.0f, -1.0f   // back top left		7
	};
	*/
	
	float vertices[] = {
		max_x, max_y, max_z,  // front top right		0
		max_x, min_y, max_z,  // front bottom right	1
		min_x, min_y, max_z,  // front bottom left	2
		min_x, max_y, max_z,   // front top left		3

		max_x, max_y, min_z,  // back top right		4
		max_x, min_y, min_z,  // back bottom right	5
		min_x, min_y, min_z,  // back bottom left	6
		min_x, max_y, min_z   // back top left		7
	};

	unsigned int indices[] = {  // note that we start from 0!
		2, 1, 0,
		3, 2, 6,
		5, 1, 0,
		4, 5, 6,
		7, 3, 7,
		4

	};
	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
	glDrawElements(GL_LINE_STRIP, 16, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);

}