#include "OpenCLSim.h"

// Error handling
const char *getErrorString(cl_int error)
{
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

void checkError(cl_int err) {
	if (err != CL_SUCCESS)
		std::cerr << getErrorString(err) << std::endl;
}




OpenCLSim::OpenCLSim() {
	cl::Platform::get(&platforms);
	platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	context = new cl::Context(devices);

	vectorAddKernel = createKernel("../CLKernels/VectorAddition.cl", "vectorAddition");
	updPosKernel = createKernel("../CLKernels/UpdPos.cl", "updPos");
}

OpenCLSim::~OpenCLSim() {
	delete context;
}


cl::Kernel OpenCLSim::createKernel(char* filepath, char* name) {
	std::ifstream helloWorldFile(filepath);
	std::string src(std::istreambuf_iterator<char>(helloWorldFile), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
	cl::Program program(*context, sources);

	cl_int err = program.build(devices, "-cl-std=CL1.2");
	checkError(err);
	
	cl::Kernel kernel(program, name, &err);
	checkError(err);


	return kernel;
}

void OpenCLSim::step() {
	

	const int N_elements = 102400;


	// Initialize data
	int* vector1 = new int[N_elements];
	int* vector2 = new int[N_elements];
	int* vector3 = new int[N_elements];

	for (size_t i = 0; i < N_elements; i++) {
		vector1[i] = i;
		vector2[i] = 2 * i;
	}

	// Allocate memory
	cl::Buffer vec1Buff(*context, CL_MEM_READ_ONLY, sizeof(int) * N_elements, vector1);
	cl::Buffer vec2Buff(*context, CL_MEM_READ_ONLY, sizeof(int) * N_elements, vector2);
	cl::Buffer vec3Buff(*context, CL_MEM_WRITE_ONLY, sizeof(int) * N_elements, vector3);
	
	// Pass arguments
	vectorAddKernel.setArg(0, vec1Buff);
	vectorAddKernel.setArg(1, vec2Buff);
	vectorAddKernel.setArg(2, vec3Buff);

	// Create command queue
	cl::CommandQueue queue(*context, devices[0]);

	// Copy data to buffers
	queue.enqueueWriteBuffer(vec1Buff, CL_TRUE, 0, sizeof(int) * N_elements, vector1);
	queue.enqueueWriteBuffer(vec2Buff, CL_TRUE, 0, sizeof(int) * N_elements, vector2);

	// Launch kernel
	queue.enqueueNDRangeKernel(vectorAddKernel, cl::NullRange, cl::NDRange(N_elements), cl::NDRange(1024));

	// Copy back data
	queue.enqueueReadBuffer(vec3Buff, CL_TRUE, 0, sizeof(int) * N_elements, vector3);


	// Print result
	std::cout << "N_elements = " << N_elements << std::endl << std::endl;
	for (size_t i = 0; i < 10; i++) {
		std::cout << vector1[i] << '+' << vector2[i] << '=' << vector3[i] << std::endl;
	}
	std::cout << "..." << std::endl;

	// Verify result
	for (size_t i = 0; i < N_elements; i++) {
		if (vector1[i] + vector2[i] != vector3[i]) {
			std::cout << "Error at index " << i << ": " << vector1[i] << '+' << vector2[i] << "!=" << vector3[i] << std::endl;
			break;
		}
	}

	std::cout << "Sucess!" << std::endl;


	// Clean up
	delete[] vector1;
	delete[] vector2;
	delete[] vector3;
	
}

void OpenCLSim::updPos(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt){

	
	cl::Buffer posBuff(*context, CL_MEM_READ_WRITE, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);
	

	// Pass arguments
	updPosKernel.setArg(0, posBuff);
	updPosKernel.setArg(1, MAX_PARTICLES);

	// Create command queue
	cl::CommandQueue queue(*context, devices[0]);

	// Copy data to buffers
	queue.enqueueWriteBuffer(posBuff, CL_TRUE, 0, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);

	// Launch kernel
	queue.enqueueNDRangeKernel(updPosKernel, cl::NullRange, cl::NDRange(MAX_PARTICLES), cl::NDRange(1024));

	// Copy back data
	queue.enqueueReadBuffer(posBuff, CL_TRUE, 0, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);
	
}