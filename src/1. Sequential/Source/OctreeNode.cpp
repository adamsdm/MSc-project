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
		children[i] = NULL;
	}
	
	usr_val = nullptr;
	no_elements = 0;
	

}

OctreeNode::~OctreeNode(){
	free();
}

void OctreeNode::free(){
	if(usr_val) delete usr_val;

	for (int i = 0; i < 8; i++){
		if (children[i])
			children[i]->free();
	}
	
	delete[] children;

}

int OctreeNode::insert(float x, float y, float z, void *usr_data){
	/*
	if (no_elements == 0){
		pos_x = x;
		pos_y = y;
		pos_z = z;
		usr_val = usr_val;
	}

	// If node already contains data, make node into a cell
	if (no_elements == 1){
		printf("Node already has a element in the cell\n");
		insert_sub(pos_x, pos_y, pos_z, usr_val);
		usr_val = NULL;
	}

	no_elements++;
	return no_elements;
	*/

	/* if this node is empty, it will be turned into a leaf by placing the
	data directly inside of it */
	if (no_elements == 0) {
		pos_x = x;
		pos_y = y;
		pos_z = z;
		usr_val = usr_val;
	}

	/* handle a node that already contains data */
	else {
		/* If this node is a leaf, take its position and place it in the
		appropriate child, no longer making this a leaf */
		if (no_elements == 1) {
			insert_sub(pos_x, pos_y, pos_z, usr_val);
			usr_val = NULL;
		}

		/* Since this node is occupied, recursively add the data to the
		appropriate child node */
		insert_sub(x, y, z, usr_data);
	}
	/* A data point was inserted into this node, therefor the element count
	must be incremented */
	no_elements++;
	return no_elements;
}



int OctreeNode::insert_sub(float x, float y, float z, void* usr_data){
	int sub = 0;

	// New boudns to be inserted into child node
	float n_min_x, n_min_y, n_min_z;
	float n_max_x, n_max_y, n_max_z;

	// Select which octant to insert the node
	/* Children 0,2,4,8 have positive x-coordinates */
	if (x > mid_x) {
		sub += 1;
		n_min_x = mid_x;
		n_max_x = max_x;
	}
	else {
		n_min_x = min_x;
		n_max_x = mid_x;
	}

	/* Children 0,1,3,4 have positive y-coordinates */
	if (y > mid_y) {
		sub += 2;
		n_min_y = mid_y;
		n_max_y = max_y;
	}
	else {
		n_min_y = min_y;
		n_max_y = mid_y;
	}
	
	/* Children 0,1,2,3 have positive z-coordinates */

	if (z > mid_z) {
		sub += 4;
		n_min_z = mid_z;
		n_max_z = max_z;
	}
	else {
		n_min_z = min_z;
		n_max_z = mid_z;
	}

	if (!children[sub])
		children[sub] = new OctreeNode(n_min_x, n_min_y, n_min_z, n_max_x, n_max_y, n_max_z);

	return children[sub]->insert(x, y, z, usr_val);
	
}

void OctreeNode::renderBounds(){
	

	for (int i = 0; i < 8; i++){
		if (children[i]){
			children[i]->renderBounds();
		}
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