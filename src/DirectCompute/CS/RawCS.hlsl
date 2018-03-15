

RWBuffer<int>		Buffer0		: register(u0);
RWBuffer<int>		Buffer1		: register(u1);
RWBuffer<int>		BufferOut   : register(u2);

[numthreads(1, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
	
	BufferOut[DTid.x] = Buffer0[DTid.x] + Buffer1[DTid.x];
}
