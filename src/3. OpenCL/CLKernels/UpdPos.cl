kernel void updPos(__global float* positions, const int MAX_PARTICLES)
{
	int i = get_global_id(0);

	if (i < MAX_PARTICLES) {
		positions[i * 3 + 0] += 10.0f;
		positions[i * 3 + 1] += 0.0f;
		positions[i * 3 + 2] += 0.0f;
	}

}