typedef struct
{
	float px, py, pz;
	float vx, vy, vz;
	float Fx, Fy, Fz;
	float weight;
} Particle;


RWBuffer<float>					positions	: register(u0);
RWStructuredBuffer<Particle>	particles	: register(u1);

[numthreads(1024, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{

	float simspeed = 0.1;
	float dt = 0.1;

	Particle p = particles[DTid.x];

	p.px += p.vx * simspeed * dt;
	p.py += p.vy * simspeed * dt;
	p.pz += p.vz * simspeed * dt;

	positions[DTid.x * 3 + 0] = p.px;
	positions[DTid.x * 3 + 1] = p.py;
	positions[DTid.x * 3 + 2] = p.pz;

	particles[DTid.x] = p;

}