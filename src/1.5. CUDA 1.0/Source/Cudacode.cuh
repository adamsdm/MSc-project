#pragma once

#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include <stdio.h>

#include <glad/glad.h>
#include "Particle.h"

void CUDAUpdatePositions(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt);