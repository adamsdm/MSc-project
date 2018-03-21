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

	// Get list of platforms
	cl::Platform::get(&platforms);


	if (!platforms.size()){
		std::cout << "No OpenCL platform found. Check installation\n";
		exit(1);
	}
	// Set default as the first platform found
	default_platform = platforms[0]; 

	// Print info
	std::cout << "----- PLATFORM -----" << std::endl
		<< default_platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl
		<< default_platform.getInfo<CL_PLATFORM_VERSION>() << std::endl << std::endl;


	// Get a list of available devices on the system
	default_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (!devices.size()){
		std::cout << " No devices found" << std::endl;
		exit(1);
	}
	// Set default device
	default_device = devices[0];

	// print device info
	std::cout << "------ DEVICE ------" << std::endl
		
		<< default_device.getInfo<CL_DEVICE_VENDOR>() << std::endl
		<< default_device.getInfo<CL_DEVICE_NAME>() << std::endl
		<< "--------------------" << std::endl << std::endl;


	// Create context from default device
	context = new cl::Context({ default_device });


	// Read and compile kernels
	vectorAddKernel = createKernel("../../OpenCL/CLKernels/VectorAddition.cl", "vectorAddition");
	updPosKernel = createKernel("../../OpenCL/CLKernels/UpdPos.cl", "updPos");
	updForceKernel = createKernel("../../OpenCL/CLKernels/UpdForce.cl", "updForce");

}

OpenCLSim::~OpenCLSim() {
	delete context;
}


cl::Kernel OpenCLSim::createKernel(char* filepath, char* name) {
	std::ifstream file(filepath);
	std::string src(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
	cl::Program program(*context, sources);

	cl_int err;
	try {
		program.build(devices, "-cl-std=CL1.2");
	}
	catch (cl::Error &e) {
		if (e.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			for (cl::Device dev : devices)
			{
				// Check the build status
				cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
				if (status != CL_BUILD_ERROR)
					continue;

				// Get the build log
				std::string name = dev.getInfo<CL_DEVICE_NAME>();
				std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				std::cout << "Build log for " << name << ":" << std::endl
					<< buildlog << std::endl;
			}
		}
		file.close();
		exit(1);
	}
	

	// Create kernel from program
	cl::Kernel kernel(program, name, &err);
	checkError(err);

	file.close();

	return kernel;
	
}


void OpenCLSim::updPos(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt){

	// Buffer containing particles
	cl::Buffer parBuff(*context, CL_MEM_READ_WRITE, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	// Buffer containing positions
	cl::Buffer posBuff(*context, CL_MEM_READ_WRITE, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);
	
	float simspeed = 0.01;

	// Pass arguments
	updPosKernel.setArg(0, posBuff);		// Positions
	updPosKernel.setArg(1, parBuff);		// Particles
	updPosKernel.setArg(2, MAX_PARTICLES);	// Max particles
	updPosKernel.setArg(3, dt);				// dt
	updPosKernel.setArg(4, simspeed);		// simspeed

	// Create command queue
	cl::CommandQueue queue(*context, default_device);

	// Copy data to buffers
	queue.enqueueWriteBuffer(parBuff, CL_TRUE, 0, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	queue.enqueueWriteBuffer(posBuff, CL_TRUE, 0, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);

	// Launch kernel
	queue.enqueueNDRangeKernel(updPosKernel, cl::NullRange, cl::NDRange(MAX_PARTICLES), cl::NDRange(1024));
	queue.finish();

	// Copy back data
	queue.enqueueReadBuffer(posBuff, CL_TRUE, 0, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);
	queue.enqueueReadBuffer(parBuff, CL_TRUE, 0, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	
}


void OpenCLSim::updFor(Particle *ParticlesContainer, sOctreeNode *nodeContainer, int count, int MAX_PARTICLES, float dt) {

	// Create buffers
	cl::Buffer parBuff(*context, CL_MEM_READ_WRITE, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	cl::Buffer nodBuff(*context, CL_MEM_READ_WRITE, count * sizeof(sOctreeNode), nodeContainer);

	// Set buffers as arguments
	updForceKernel.setArg(0, parBuff);			// Particles
	updForceKernel.setArg(1, nodBuff);			// Nodes
	updForceKernel.setArg(2, MAX_PARTICLES);	// MAX_PARTICLES
	updForceKernel.setArg(3, count);	// MAX_PARTICLES

	// Create command queue and copy data
	cl::CommandQueue queue(*context, default_device);
	queue.enqueueWriteBuffer(parBuff, CL_TRUE, 0, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	queue.enqueueWriteBuffer(nodBuff, CL_TRUE, 0, count * sizeof(sOctreeNode), nodeContainer);

	// Launch kernel
	queue.enqueueNDRangeKernel(updForceKernel, cl::NullRange, cl::NDRange(MAX_PARTICLES), cl::NDRange(1024));
	queue.finish();

	// Read back result
	queue.enqueueReadBuffer(parBuff, CL_TRUE, 0, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	
}

void OpenCLSim::step(Particle *ParticlesContainer, sOctreeNode *nodeContainer, GLfloat *g_particule_position_size_data, int count, unsigned int MAX_PARTICLES, float dt){
	
	// Create buffers
	cl::Buffer parBuff(*context, CL_MEM_READ_WRITE, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	cl::Buffer nodBuff(*context, CL_MEM_READ_ONLY, count * sizeof(sOctreeNode), nodeContainer);
	cl::Buffer posBuff(*context, CL_MEM_READ_WRITE, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);


	float simspeed = 0.01;

	// Set updateForce kernel arguments
	updForceKernel.setArg(0, parBuff);			// Particles
	updForceKernel.setArg(1, nodBuff);			// Nodes
	updForceKernel.setArg(2, MAX_PARTICLES);	// MAX_PARTICLES
	updForceKernel.setArg(3, count);			// MAX_PARTICLES

	

	// Set update position kernel arguments
	updPosKernel.setArg(0, posBuff);		// Positions
	updPosKernel.setArg(1, parBuff);		// Particles
	updPosKernel.setArg(2, MAX_PARTICLES);	// Max particles
	updPosKernel.setArg(3, dt);				// dt
	updPosKernel.setArg(4, simspeed);		// simspeed


	// Create command queue and copy data
	cl::CommandQueue queue(*context, default_device);
	queue.enqueueWriteBuffer(parBuff, CL_TRUE, 0, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);
	queue.enqueueWriteBuffer(nodBuff, CL_TRUE, 0, count * sizeof(sOctreeNode), nodeContainer);
	queue.enqueueWriteBuffer(posBuff, CL_TRUE, 0, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);


	// Launch kernels
	queue.enqueueNDRangeKernel(updForceKernel, cl::NullRange, cl::NDRange(MAX_PARTICLES), cl::NDRange(1024));
	queue.finish();
	queue.enqueueNDRangeKernel(updPosKernel, cl::NullRange, cl::NDRange(MAX_PARTICLES), cl::NDRange(1024));
	queue.finish();

	// Copy back data
	queue.enqueueReadBuffer(posBuff, CL_TRUE, 0, 3 * MAX_PARTICLES * sizeof(GLfloat), g_particule_position_size_data);
	queue.enqueueReadBuffer(parBuff, CL_TRUE, 0, MAX_PARTICLES * sizeof(Particle), ParticlesContainer);

}