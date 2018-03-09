

RWBuffer<float>		pos		: register(u0);

[numthreads(1024, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
	pos[3 * DTid.x + 0] = DTid.x;
	pos[3 * DTid.x + 1] = 0.0f;
	pos[3 * DTid.x + 2] = 100.0f;

}