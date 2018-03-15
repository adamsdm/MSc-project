#ifndef M_PI
#define M_PI	3.14159265359
#endif

#ifndef G
#define G		6.672e-11F
#endif

#ifndef SOFTENING
#define SOFTENING 1e-9f
#endif

typedef struct
{
	float px, py, pz;
	float vx, vy, vz;
	float Fx, Fy, Fz;
	float weight;
} Particle;



typedef struct {
	int index;
	unsigned int childIndices[8];

	float min_x, min_y, min_z;
	float max_x, max_y, max_z;

	float mid_x, mid_y, mid_z;

	float m;
	float com_x, com_y, com_z;


} sOctreeNode;



kernel void updForce(__global Particle* particles, __global sOctreeNode* nodes, int MAX_PARTICLES, int count)
{
	int idx = get_global_id(0);


	if (idx < MAX_PARTICLES) {
		Particle p = particles[idx];

		// Queue variables
		const int MAX_SIZE = 10000;
		int f = -1, r = -1;
		int queue[10000];


		// PUSH
		if ((r + 1) % MAX_SIZE == f) { printf("Error: Queue is Full\n"); return; } // Is Full?
		if (f == -1 && r == -1) { f = r = 0; }		// Is empty?
		else { r = (r + 1) % MAX_SIZE;}
		queue[r] = nodes[0].index;

		while (!(f == -1 && r == -1)) {
			sOctreeNode node = nodes[queue[f]];
			// POP
			{
				if (f == -1 && r == -1) { ; } // Is empty?
				else if (f == r) { r = f = -1; }
				else { f = (f + 1) % MAX_SIZE; }
			}

			float dx = node.com_x - p.px;
			float dy = node.com_y - p.py;
			float dz = node.com_z - p.pz;

			float dist = sqrt(dx*dx + dy*dy + dz*dz);

			if (dist == 0) continue;

			float width = ((node.max_x - node.min_x) +
				(node.max_y - node.min_y) +
				(node.max_z - node.min_z)) / 3;

			// The node is far away enough to be evaluated as a single node
			if (width / dist < 0.5) {

				float F = (6.672e-11F * p.weight * node.m) / (dist + 1.0f + 1e-9f * 1e-9f);

				p.vx += F * dx / dist;
				p.vy += F * dy / dist;
				p.vz += F * dz / dist;
				
				
				
			}
			else {
				for (int i = 0; i < 8; i++){
					if (node.childIndices[i]){
						// PUSH
							{
								if ((r + 1) % MAX_SIZE == f) { printf("Error: Queue is Full\n"); return; } // Is Full?
								if (f == -1 && r == -1) { f = r = 0; }		// Is empty?
								else { r = (r + 1) % MAX_SIZE; }
								queue[r] = node.childIndices[i];
							}
					}
				}
			}
		}

		particles[idx] = p;

	}
}