typedef struct
{
	float px, py, pz;
	float vx, vy, vz;
	float Fx, Fy, Fz;
	float weight;
} Particle;



kernel void updPos(__global float* positions, __global Particle* particles, int MAX_PARTICLES, float dt, float simspeed)
{

	int i = get_global_id(0);


	if (i < MAX_PARTICLES) {

		Particle p = particles[i];
		
		p.px += p.vx * simspeed * dt;
		p.py += p.vy * simspeed * dt;
		p.pz += p.vz * simspeed * dt;

		positions[i * 3 + 0] = p.px;
		positions[i * 3 + 1] = p.py;
		positions[i * 3 + 2] = p.pz;

		particles[i] = p;
	}

}