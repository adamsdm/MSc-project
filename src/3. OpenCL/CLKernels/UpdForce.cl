typedef struct
{
	float px, py, pz;
	float vx, vy, vz;
	float Fx, Fy, Fz;
	float weight;
} Particle;



typedef struct {
	unsigned int childIndices[8];

	float min_x, min_y, min_z;
	float max_x, max_y, max_z;

	float mid_x, mid_y, mid_z;

	float m;
	float com_x, com_y, com_z;


} sOctreeNode;


/*
struct MyQueue {
	static const int MAX_SIZE;
	int f, r;
	int A[10000];	// TODO: More dynamic array size


	bool empty() {
		return (f == -1 && r == -1);
	};

	bool isFull() {
		return (r + 1) % MAX_SIZE == f ? true : false;
	}

	void push(int x)
	{
		if (isFull())
		{
			printf("Error: Queue is Full\n");
			return;
		}
		if (empty())
		{
			f = r = 0;
		}
		else
		{
			r = (r + 1) % MAX_SIZE;
		}
		A[r] = x;
	}

	void pop()
	{
		if (empty())
		{
			printf("Error: Queue is Empty\n");
			return;
		}
		else if (f == r)
		{
			r = f = -1;
		}
		else
		{
			f = (f + 1) % MAX_SIZE;
		}
	}

	int front()
	{
		if (f == -1)
		{
			printf("Error: cannot return front from empty queue\n");
			return -1;
		}
		return A[f];
	}

	void print()
	{
		// Finding number of elements in queue  
		int count = (r + MAX_SIZE - f) % MAX_SIZE + 1;
		printf("Queue       : ");
		for (int i = 0; i <count; i++)
		{
			int index = (f + i) % MAX_SIZE; // Index of element while travesing circularly from front
			printf("%d ", A[index]);
		}
		printf("\n\n");
	}
};
*/




kernel void updForce(__global Particle* particles, __global sOctreeNode* nodes, int MAX_PARTICLES)
{
	int i = get_global_id(0);

	if (i < MAX_PARTICLES) {
		Particle p = particles[i];

		p.vx = 5000.0f;

		particles[i] = p;
	}
}