
#include "DCSim.h"

DCSim::DCSim(){

}

DCSim::~DCSim() {

}

void DCSim::step(Particle *ParticlesContainer, sOctreeNode *nodeContainer, GLfloat *g_particule_position_size_data, int count, unsigned int MAX_PARTICLES, float dt){
	std::cout << "DC STEP" << std::endl;
}

HRESULT DCSim::CreateComputeDevice(ID3D11Device** ppDeviceOut, ID3D11DeviceContext** ppContextOut, bool bForceRef){
	*ppDeviceOut = nullptr;
	*ppContextOut = nullptr;

	HRESULT hr = S_OK;

	UINT uCreationFlags = D3D11_CREATE_DEVICE_SINGLETHREADED;
	D3D_FEATURE_LEVEL flOut;
	static const D3D_FEATURE_LEVEL flvl[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };

	bool bNeedRefDevice = false;
	if (!bForceRef)
	{
		hr = D3D11CreateDevice(nullptr,     // Use default graphics card
			D3D_DRIVER_TYPE_HARDWARE,		// Try to create a hardware accelerated device
			nullptr,                        // Do not use external software rasterizer module
			uCreationFlags,					// Device creation flags
			flvl,
			sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
			D3D11_SDK_VERSION,           // SDK version
			ppDeviceOut,                 // Device out
			&flOut,                      // Actual feature level created
			ppContextOut);              // Context out

		if (SUCCEEDED(hr))
		{
			// A hardware accelerated device has been created, so check for Compute Shader support

			// If we have a device >= D3D_FEATURE_LEVEL_11_0 created, full CS5.0 support is guaranteed, no need for further checks
			if (flOut < D3D_FEATURE_LEVEL_11_0)
			{
				// Otherwise, we need further check whether this device support CS4.x (Compute on 10)
				D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts;
				(*ppDeviceOut)->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts));
				if (!hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x)
				{
					bNeedRefDevice = true;
					printf("No hardware Compute Shader capable device found, trying to create ref device.\n");
				}
			}
		}
	}

	if (bForceRef || FAILED(hr) || bNeedRefDevice)
	{
		// Either because of failure on creating a hardware device or hardware lacking CS capability, we create a ref device here

		SAFE_RELEASE(*ppDeviceOut);
		SAFE_RELEASE(*ppContextOut);

		hr = D3D11CreateDevice(nullptr,                        // Use default graphics card
			D3D_DRIVER_TYPE_REFERENCE,   // Try to create a hardware accelerated device
			nullptr,                        // Do not use external software rasterizer module
			uCreationFlags,              // Device creation flags
			flvl,
			sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
			D3D11_SDK_VERSION,           // SDK version
			ppDeviceOut,                 // Device out
			&flOut,                      // Actual feature level created
			ppContextOut);              // Context out
		if (FAILED(hr))
		{
			printf("Reference rasterizer device create failure\n");
			return hr;
		}
	}

	return hr;
}