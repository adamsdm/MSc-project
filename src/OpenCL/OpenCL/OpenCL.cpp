# define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <exception>

#define N_elements	102400

int main()
{	
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::vector<cl::Device> devices;
	
	platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

	std::ifstream helloWorldFile("VectorAddition.cl");
	std::string src(std::istreambuf_iterator<char>(helloWorldFile), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
	cl::Context context(devices);
	cl::Program program(context, sources);

	cl_int err = program.build(devices, "-cl-std=CL1.2");



	int* vector1 = new int[N_elements];
	int* vector2 = new int[N_elements];

	for (size_t i = 0; i < N_elements; i++){
		vector1[i] = i;
		vector2[i] = 2*i;
	}
	

	cl::Buffer vec1Buff(context, CL_MEM_READ_ONLY, sizeof(int) * N_elements, vector1);
	cl::Buffer vec2Buff(context, CL_MEM_READ_ONLY, sizeof(int) * N_elements, vector2);

	int* vector3 = new int[N_elements];
	cl::Buffer vec3Buff(context, CL_MEM_WRITE_ONLY, sizeof(int) * N_elements, vector3);

	cl::Kernel kernel(program, "vectorAddition", &err);
	kernel.setArg(0, vec1Buff);
	kernel.setArg(1, vec2Buff);
	kernel.setArg(2, vec3Buff);

	cl::CommandQueue queue(context, devices[0]);
	queue.enqueueWriteBuffer(vec1Buff, CL_TRUE, 0, sizeof(int) * N_elements, vector1);
	queue.enqueueWriteBuffer(vec2Buff, CL_TRUE, 0, sizeof(int) * N_elements, vector2);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N_elements), cl::NDRange(1024));

	queue.enqueueReadBuffer(vec3Buff, CL_TRUE, 0, sizeof(int) * N_elements, vector3);

	
	// Print result
	std::cout << "N_elements = " << N_elements << std::endl << std::endl;
	for (size_t i = 0; i < 10; i++){
		std::cout << vector1[i] << '+' << vector2[i] << '=' << vector3[i] << std::endl;
	}
	std::cout << "..." << std::endl;
	
	// Verify result
	for (size_t i = 0; i < N_elements; i++){
		if (vector1[i] + vector2[i] != vector3[i]) {
			std::cout << "Error at index " << i << ": " << vector1[i] << '+' << vector2[i] << "!=" << vector3[i] << std::endl;
			return EXIT_FAILURE;
		}		
	}

	std::cout << "Sucess!" << std::endl;
		

	// Clean up
	delete[] vector1;
	delete[] vector2;
	delete[] vector3;
	
	
	return EXIT_SUCCESS;
}