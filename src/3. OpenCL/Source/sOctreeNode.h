#pragma once

typedef struct {
	unsigned int childIndices[8];

	float min_x, min_y, min_z;
	float max_x, max_y, max_z;

	float mid_x, mid_y, mid_z;

	float m;
	float com_x, com_y, com_z;


} sOctreeNode;
