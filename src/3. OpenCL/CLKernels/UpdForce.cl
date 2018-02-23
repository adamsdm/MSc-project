typedef struct
{
	float px, py, pz;
	float vx, vy, vz;
	float Fx, Fy, Fz;
	float weight;
} Particle;


kernel void updForce(__global Particle* particles, int MAX_PARTICLES)
{
	int i = get_global_id(0);
	
	if (i < MAX_PARTICLES) {
		Particle p = particles[i];

		p.vx = 1000.0f;

		particles[i] = p;
	}
}