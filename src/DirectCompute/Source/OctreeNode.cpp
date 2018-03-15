// Implementation of a Octree which can contain both cells and particles (leafs)

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
	
	m = NULL;
	com_x = NULL;
	com_y = NULL;
	com_z = NULL;

	no_elements = 0;
}

OctreeNode::~OctreeNode(){
	freeTree();
}

void OctreeNode::freeTree(){
	
	for (int i = 0; i < 8; i++){
		if (children[i])
			children[i]->freeTree();
	}

	free(children);
}

int OctreeNode::insert(float x, float y, float z, float pmass, float pcom_x, float pcom_y, float pcom_z){

	/* if this node is empty, it will be turned into a leaf by placing the
	data directly inside of it */
	if (no_elements == 0) {
		pos_x = x;
		pos_y = y;
		pos_z = z;

		m = pmass;
		com_x = pcom_x;
		com_y = pcom_y;
		com_z = pcom_z;
	}

	/* handle a node that already contains data */
	else {
		/* If this node is a leaf, take its position and place it in the
		appropriate child, no longer making this a leaf */
		if (no_elements == 1) {
			insert_sub(pos_x, pos_y, pos_z, m, com_x, com_y, com_z);
			m = NULL;
			com_x = NULL;
			com_y = NULL;
			com_z = NULL;
		}

		/* Since this node is occupied, recursively add the data to the
		appropriate child node */
		insert_sub(x, y, z, pmass, pcom_x, pcom_y, pcom_z);
	}
	/* A data point was inserted into this node, therefor the element count
	must be incremented */
	no_elements++;
	return no_elements;
}

int OctreeNode::insert_sub(float x, float y, float z, float pmass, float pcom_x, float pcom_y, float pcom_z){
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

	return children[sub]->insert(x, y, z, pmass, pcom_x, pcom_y, pcom_z);
	
}

void OctreeNode::setModelAndRender(Shader boxShader){

	for (int i = 0; i < 8; i++){
		if (children[i])
			children[i]->setModelAndRender(boxShader);
	}


	glm::vec3 size = 0.5f*glm::vec3(max_x - min_x, max_y - min_y, max_z - min_z);
	glm::vec3 center = glm::vec3(mid_x, mid_y, mid_z);
	glm::mat4 model = glm::translate(glm::mat4(1), center) * glm::scale(glm::mat4(1), size);


	boxShader.setMat4("model", model);
	glDrawElements(GL_LINE_STRIP, 16, GL_UNSIGNED_INT, 0);

}