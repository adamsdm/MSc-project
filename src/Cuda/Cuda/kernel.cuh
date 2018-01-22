#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

__global__ void add(int *out, const int *in_a, const int *in_b);