
#include "DCSim.h"

#define NUM_ELEMENTS	1024


struct BufType
{
	int i;
	float f;
};


DCSim::DCSim(){

	// Create a device that the kernel will run on
	std::cout << "Creating compute device... ";
	if (FAILED(CreateComputeDevice(&device, &context, false))) exit(1);
	std::cout << "Done!" << std::endl;

	// Compile CS from source
	std::cout << "Creating Compute Shaders...";
	if (FAILED(CreateComputeShader(L"../Shaders/UpdPos.hlsl", "CSMain", device, &updPosCS))) exit(1);
	std::cout << " Done!" << std::endl;

	// ================================= //
	// EX 1. Raw Buffers vector addition //
	// ================================= //
	/*
	// Create a device that the kernel will run on
	std::cout << "Creating compute device... ";
	if (FAILED(CreateComputeDevice(&device, &context, false))) exit(1);
	std::cout << "Done!" << std::endl;


	// Compile CS from source
	std::cout << "Creating Compute Shaders...";
	if (FAILED(CreateComputeShader(L"../Shaders/UpdPos.hlsl", "CSMain", device, &updPosCS))) exit(1);
	std::cout << " Done!" << std::endl;

	// Create input data and buffers
	std::cout << "Creating buffers and filling them with initial data...";

	int g_vBuf0[NUM_ELEMENTS];
	int g_vBuf1[NUM_ELEMENTS];
	int g_vBufRes[NUM_ELEMENTS];

	for (int i = 0; i < NUM_ELEMENTS; ++i)
	{
		g_vBuf0[i] = i;
		g_vBuf1[i] = 3*i;
	}

	std::cout << " Done!" << std::endl;

	ID3D11Buffer* g_pBuf0 = nullptr;
	ID3D11Buffer* g_pBuf1 = nullptr;
	ID3D11Buffer* g_pBufResult = nullptr;


	// Create raws buffer from input data
	CreateRawBuffer(device, sizeof(int), NUM_ELEMENTS, &g_vBuf0[0], &g_pBuf0);
	CreateRawBuffer(device, sizeof(int), NUM_ELEMENTS, &g_vBuf1[0], &g_pBuf1);
	CreateRawBuffer(device, sizeof(int), NUM_ELEMENTS, &g_vBufRes[0], &g_pBufResult);

	// Create acces views that can be handled by the CS
	ID3D11UnorderedAccessView*  g_pBuf0UAV = nullptr;
	ID3D11UnorderedAccessView*  g_pBuf1UAV = nullptr;
	ID3D11UnorderedAccessView*	g_pBufResultUAV = nullptr;

	// Creating shader resource views for reading and unordered access views for writing
	std::cout << "Creating SRV and UAV for writing...";
	CreateBufferUAV(device, g_pBuf0, &g_pBuf0UAV);
	CreateBufferUAV(device, g_pBuf1, &g_pBuf1UAV);
	CreateBufferUAV(device, g_pBufResult, &g_pBufResultUAV);
	std::cout << " Done!" << std::endl;

	printf("Running Compute Shader...");

	{
		context->CSSetShader(updPosCS, nullptr, 0);
		context->CSSetUnorderedAccessViews(0, 1, &g_pBuf0UAV, nullptr);
		context->CSSetUnorderedAccessViews(1, 1, &g_pBuf1UAV, nullptr);
		context->CSSetUnorderedAccessViews(2, 1, &g_pBufResultUAV, nullptr);
		context->Dispatch(NUM_ELEMENTS, 1, 1);



		// Unbind shader and acces views
		context->CSSetShader(nullptr, nullptr, 0);	
		ID3D11UnorderedAccessView* ppUAViewnullptr[1] = { nullptr };
		context->CSSetUnorderedAccessViews(0, 1, ppUAViewnullptr, nullptr);
	}

	printf("done\n");

	// Read back the result from GPU, verify its correctness against result computed by CPU
	{
		ID3D11Buffer* debugbuf = CreateAndCopyToDebugBuf(device, context, g_pBufResult);
		D3D11_MAPPED_SUBRESOURCE MappedResource;
		context->Map(debugbuf, 0, D3D11_MAP_READ, 0, &MappedResource);

		// Set a break point here and put down the expression "p, 1024" in your watch window to see what has been written out by our CS
		// This is also a common trick to debug CS programs.
		int *p = (int*)MappedResource.pData;

		// Verify that if Compute Shader has done right
		printf("Verifying against CPU result...");
		bool bSuccess = true;

		//Print first 10 results
		for (size_t i = 0; i < 10; i++) {
			printf("%d + %d = %d \n", g_vBuf0[i], g_vBuf1[i], p[i]);
		}

		context->Unmap(debugbuf, 0);
		SAFE_RELEASE(debugbuf);
	}
	*/





	// ======================================== //
	// EX 2. Structured Buffers vector addition //
	// ======================================== //
	
	/*
	// Create a device that the kernel will run on
	std::cout << "Creating compute device... ";
	if (FAILED(CreateComputeDevice(&device, &context, false))) exit(1);
	std::cout << "Done!" << std::endl;


	// Compile CS from source
	std::cout << "Creating Compute Shaders...";
	if (FAILED(CreateComputeShader(L"../Shaders/VectorAddCS.hlsl", "CSMain", device, &vecAddCS))) exit(1);
	std::cout << " Done!" << std::endl;

	// Create input data and buffers
	std::cout << "Creating buffers and filling them with initial data...";

	BufType g_vBuf0[NUM_ELEMENTS];
	BufType g_vBuf1[NUM_ELEMENTS];
	for (int i = 0; i < NUM_ELEMENTS; ++i)
	{
		g_vBuf0[i].i = i;
		g_vBuf0[i].f = (float)i + i/10.0f;

		g_vBuf1[i].i = 2 * i;
		g_vBuf1[i].f = 2.0f*(float)i + i/10.0f;
	}

	std::cout << " Done!" << std::endl;

	ID3D11Buffer* g_pBuf0 = nullptr;
	ID3D11Buffer* g_pBuf1 = nullptr;
	ID3D11Buffer* g_pBufResult = nullptr;


	
	// Create structured buffer from input data
	CreateStructuredBuffer(device, sizeof(BufType), NUM_ELEMENTS, &g_vBuf0[0], &g_pBuf0);
	CreateStructuredBuffer(device, sizeof(BufType), NUM_ELEMENTS, &g_vBuf1[0], &g_pBuf1);
	CreateStructuredBuffer(device, sizeof(BufType), NUM_ELEMENTS, nullptr, &g_pBufResult);


	ID3D11ShaderResourceView*   g_pBuf0SRV = nullptr;
	ID3D11ShaderResourceView*   g_pBuf1SRV = nullptr;
	ID3D11UnorderedAccessView*  g_pBufResultUAV = nullptr;

	// Creating shader resource views for reading and unordered access views for writing
	std::cout << "Creating SRV and UAV for writing...";
	CreateBufferSRV(device, g_pBuf0, &g_pBuf0SRV);
	CreateBufferSRV(device, g_pBuf1, &g_pBuf1SRV);
	CreateBufferUAV(device, g_pBufResult, &g_pBufResultUAV);
	std::cout << " Done!" << std::endl;

	printf("Running Compute Shader...");
	

	{
		context->CSSetShader(vecAddCS, nullptr, 0);
		
		ID3D11ShaderResourceView* aRViews[2] = { g_pBuf0SRV, g_pBuf1SRV };
		context->CSSetShaderResources(0, 2, aRViews);

		context->CSSetUnorderedAccessViews(0, 1, &g_pBufResultUAV, nullptr);
		context->Dispatch(NUM_ELEMENTS, 1, 1);

		context->CSSetShader(nullptr, nullptr, 0);

		ID3D11UnorderedAccessView* ppUAViewnullptr[1] = { nullptr };
		context->CSSetUnorderedAccessViews(0, 1, ppUAViewnullptr, nullptr);

		ID3D11ShaderResourceView* ppSRVnullptr[2] = { nullptr, nullptr };
		context->CSSetShaderResources(0, 2, ppSRVnullptr);

		ID3D11Buffer* ppCBnullptr[1] = { nullptr };
		context->CSSetConstantBuffers(0, 1, ppCBnullptr);
	}

	printf("done\n");

	
	// Read back the result from GPU, verify its correctness against result computed by CPU
	{
		ID3D11Buffer* debugbuf = CreateAndCopyToDebugBuf(device, context, g_pBufResult);
		D3D11_MAPPED_SUBRESOURCE MappedResource;
		BufType *p;
		context->Map(debugbuf, 0, D3D11_MAP_READ, 0, &MappedResource);

		// Set a break point here and put down the expression "p, 1024" in your watch window to see what has been written out by our CS
		// This is also a common trick to debug CS programs.
		p = (BufType*)MappedResource.pData;

		// Verify that if Compute Shader has done right
		printf("Verifying against CPU result...");
		bool bSuccess = true;

		//Print first 10 results
		for (size_t i = 0; i < 10; i++) {
			//printf("%d + %d = %d \n", g_vBuf0[i].i, g_vBuf1[i].i, p[i].i);
			printf("%.1f + %.1f = %.1f \n", g_vBuf0[i].f, g_vBuf1[i].f, p[i].f);
		}

		context->Unmap(debugbuf, 0);
		SAFE_RELEASE(debugbuf);
	}
	
	
	exit(1);
	*/
	
}

DCSim::~DCSim() {
	SAFE_RELEASE(device);
	SAFE_RELEASE(context);
	SAFE_RELEASE(vecAddCS);
}

void DCSim::updPos(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt){

	ID3D11Buffer* g_pos_buf = nullptr;
	ID3D11UnorderedAccessView*  g_pos_bufUAV = nullptr;
	

	// Create structured buffer from input data
	CreateRawBuffer(device, sizeof(GLfloat), 3 * MAX_PARTICLES, &g_particule_position_size_data[0], &g_pos_buf);

	

	// Creating shader resource views for reading and unordered access views for writing
	CreateBufferUAV(device, g_pos_buf, &g_pos_bufUAV);
	

	// Run shader
	{
		context->CSSetShader(updPosCS, nullptr, 0);
		context->CSSetUnorderedAccessViews(0, 1, &g_pos_bufUAV, nullptr);
		context->Dispatch(MAX_PARTICLES/1024, 1, 1);
		context->CSSetShader(nullptr, nullptr, 0);

		ID3D11UnorderedAccessView* ppUAViewnullptr[1] = { nullptr };
		context->CSSetUnorderedAccessViews(0, 1, ppUAViewnullptr, nullptr);

		ID3D11ShaderResourceView* ppSRVnullptr[2] = { nullptr, nullptr };
		context->CSSetShaderResources(0, 2, ppSRVnullptr);

		ID3D11Buffer* ppCBnullptr[1] = { nullptr };
		context->CSSetConstantBuffers(0, 1, ppCBnullptr);
	}

	// Read back the result from GPU, verify its correctness against result computed by CPU
	{
		ID3D11Buffer* debugbuf = CreateAndCopyToDebugBuf(device, context, g_pos_buf);
		D3D11_MAPPED_SUBRESOURCE MappedResource;
		context->Map(debugbuf, 0, D3D11_MAP_READ, 0, &MappedResource);

		// Set a break point here and put down the expression "p, 1024" in your watch window to see what has been written out by our CS
		// This is also a common trick to debug CS programs.
		
		memcpy(g_particule_position_size_data, (GLfloat*)MappedResource.pData, 3 * MAX_PARTICLES*sizeof(GLfloat));
		

		context->Unmap(debugbuf, 0);	
		SAFE_RELEASE(debugbuf);
	}
	

	SAFE_RELEASE(g_pos_buf);
	
}

void DCSim::step(Particle *ParticlesContainer, sOctreeNode *nodeContainer, GLfloat *g_particule_position_size_data, int count, unsigned int MAX_PARTICLES, float dt){
	std::cout << "DC STEP" << std::endl;
}

HRESULT DCSim::CreateComputeDevice(ID3D11Device** deviceOut, ID3D11DeviceContext** contextOut, bool bForceRef){
	*deviceOut = nullptr;
	*contextOut = nullptr;

	// Status of 
	HRESULT hr = S_OK;

	// We will only call methods of Direct3D 11 interfaces from a single thread.
	UINT flags = D3D11_CREATE_DEVICE_SINGLETHREADED;
	D3D_FEATURE_LEVEL featureLevelOut;
	static const D3D_FEATURE_LEVEL flvl[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };

	bool bNeedRefDevice = false;
	if (!bForceRef)
	{
		hr = D3D11CreateDevice(nullptr,     // Use default graphics card
			D3D_DRIVER_TYPE_HARDWARE,		// Try to create a hardware accelerated device
			nullptr,                        // Do not use external software rasterizer module
			flags,							// Device creation flags
			flvl,
			sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
			D3D11_SDK_VERSION,				// SDK version
			deviceOut,						// Device out
			&featureLevelOut,               // Actual feature level created
			contextOut);					// Context out

		if (SUCCEEDED(hr))
		{
			// A hardware accelerated device has been created, so check for Compute Shader support

			// If we have a device >= D3D_FEATURE_LEVEL_11_0 created, full CS5.0 support is guaranteed, no need for further checks
			if (featureLevelOut < D3D_FEATURE_LEVEL_11_0)
			{
				// Otherwise, we need further check whether this device support CS4.x (Compute on 10)
				D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts;
				(*deviceOut)->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts));
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
		SAFE_RELEASE(*deviceOut);
		SAFE_RELEASE(*contextOut);

		hr = D3D11CreateDevice(nullptr,				// Use default graphics card
			D3D_DRIVER_TYPE_REFERENCE,				// Try to create a hardware accelerated device
			nullptr,								// Do not use external software rasterizer module
			flags,									// Device creation flags
			flvl,
			sizeof(flvl) / sizeof(D3D_FEATURE_LEVEL),
			D3D11_SDK_VERSION,						// SDK version
			deviceOut,								// Device out
			&featureLevelOut,						// Actual feature level created
			contextOut);							// Context out
		if (FAILED(hr))
		{
			printf("Reference rasterizer device create failure\n");
			return hr;
		}
	}

	return hr;
}

HRESULT DCSim::CreateComputeShader(LPCWSTR pSrcFile, LPCSTR pFunctionName, ID3D11Device* pDevice, ID3D11ComputeShader** ppShaderOut)
{
	HRESULT hr = S_OK;

	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;

	const D3D_SHADER_MACRO defines[] =
	{
		"USE_STRUCTURED_BUFFERS", "1",
		nullptr, nullptr
	};


	// We generally prefer to use the higher CS shader profile when possible as CS 5.0 is better performance on 11-class hardware
	LPCSTR pProfile = (pDevice->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0) ? "cs_5_0" : "cs_4_0";


	ID3DBlob* pErrorBlob = nullptr;
	ID3DBlob* pBlob = nullptr;

#if D3D_COMPILER_VERSION >= 46
	hr = D3DCompileFromFile(pSrcFile, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, pFunctionName, pProfile, dwShaderFlags, 0, &pBlob, &pErrorBlob);
#else
	hr = D3DX11CompileFromFile(pSrcFile, defines, nullptr, pFunctionName, pProfile, dwShaderFlags, 0, nullptr, &pBlob, &pErrorBlob, nullptr);
#endif

	if (FAILED(hr))
	{
		if (pErrorBlob){
			std::cout << (char*)pErrorBlob->GetBufferPointer() << std::endl;
		}


		SAFE_RELEASE(pErrorBlob);
		SAFE_RELEASE(pBlob);

		return hr;
	}

	

	hr = pDevice->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, ppShaderOut);

	SAFE_RELEASE(pErrorBlob);
	SAFE_RELEASE(pBlob);


	return hr;
}

HRESULT DCSim::CreateRawBuffer(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, void* pInitData, ID3D11Buffer** ppBufOut){
	*ppBufOut = nullptr;

	D3D11_BUFFER_DESC desc;
	ZeroMemory(&desc, sizeof(desc));

	desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	desc.ByteWidth = uElementSize * uCount;
	desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
	desc.StructureByteStride = uElementSize;

	if (pInitData)
	{
		D3D11_SUBRESOURCE_DATA InitData;
		InitData.pSysMem = pInitData;
		return pDevice->CreateBuffer(&desc, &InitData, ppBufOut);
	}
	else
		return pDevice->CreateBuffer(&desc, nullptr, ppBufOut);
}

HRESULT DCSim::CreateStructuredBuffer(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, void* pInitData, ID3D11Buffer** ppBufOut)
{
	*ppBufOut = nullptr;

	D3D11_BUFFER_DESC desc;
	ZeroMemory(&desc, sizeof(desc));

	

	desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
	desc.ByteWidth = uElementSize * uCount;
	desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	desc.StructureByteStride = uElementSize;

	if (pInitData)
	{
		D3D11_SUBRESOURCE_DATA InitData;
		InitData.pSysMem = pInitData;
		return pDevice->CreateBuffer(&desc, &InitData, ppBufOut);
	}
	else
		return pDevice->CreateBuffer(&desc, nullptr, ppBufOut);
}

HRESULT DCSim::CreateBufferSRV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11ShaderResourceView** ppSRVOut)
{
	D3D11_BUFFER_DESC descBuf;
	ZeroMemory(&descBuf, sizeof(descBuf));
	pBuffer->GetDesc(&descBuf);

	D3D11_SHADER_RESOURCE_VIEW_DESC desc;
	ZeroMemory(&desc, sizeof(desc));
	desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
	desc.BufferEx.FirstElement = 0;

	if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
	{
		// This is a Raw Buffer

		desc.Format = DXGI_FORMAT_R32_TYPELESS;
		desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
		desc.BufferEx.NumElements = descBuf.ByteWidth / 4;
	}
	else
		if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)
		{
		// This is a Structured Buffer

		desc.Format = DXGI_FORMAT_UNKNOWN;
		desc.BufferEx.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
		}
		else
		{
			return E_INVALIDARG;
		}

	return pDevice->CreateShaderResourceView(pBuffer, &desc, ppSRVOut);
}

HRESULT DCSim::CreateBufferUAV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11UnorderedAccessView** ppUAVOut)
{
	D3D11_BUFFER_DESC descBuf;
	ZeroMemory(&descBuf, sizeof(descBuf));
	pBuffer->GetDesc(&descBuf);

	D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
	ZeroMemory(&desc, sizeof(desc));
	desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	desc.Buffer.FirstElement = 0;

	if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
	{
		// This is a Raw Buffer

		desc.Format = DXGI_FORMAT_R32_TYPELESS; // Format must be DXGI_FORMAT_R32_TYPELESS, when creating Raw Unordered Access View
		desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
		desc.Buffer.NumElements = descBuf.ByteWidth / 4;
	}
	else
		if (descBuf.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)
		{
		// This is a Structured Buffer

		desc.Format = DXGI_FORMAT_UNKNOWN;      // Format must be must be DXGI_FORMAT_UNKNOWN, when creating a View of a Structured Buffer
		desc.Buffer.NumElements = descBuf.ByteWidth / descBuf.StructureByteStride;
		}
		else
		{
			return E_INVALIDARG;
		}

	return pDevice->CreateUnorderedAccessView(pBuffer, &desc, ppUAVOut);
}

void DCSim::RunComputeShader(ID3D11DeviceContext* pd3dImmediateContext,
	ID3D11ComputeShader* pComputeShader,
	UINT nNumViews, ID3D11ShaderResourceView** pShaderResourceViews,
	ID3D11Buffer* pCBCS, void* pCSData, DWORD dwNumDataBytes,
	ID3D11UnorderedAccessView* pUnorderedAccessView,
	UINT X, UINT Y, UINT Z)
{
	pd3dImmediateContext->CSSetShader(pComputeShader, nullptr, 0);
	pd3dImmediateContext->CSSetShaderResources(0, nNumViews, pShaderResourceViews);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &pUnorderedAccessView, nullptr);
	if (pCBCS && pCSData)
	{
		D3D11_MAPPED_SUBRESOURCE MappedResource;
		pd3dImmediateContext->Map(pCBCS, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
		memcpy(MappedResource.pData, pCSData, dwNumDataBytes);
		pd3dImmediateContext->Unmap(pCBCS, 0);
		ID3D11Buffer* ppCB[1] = { pCBCS };
		pd3dImmediateContext->CSSetConstantBuffers(0, 1, ppCB);
	}

	pd3dImmediateContext->Dispatch(X, Y, Z);

	pd3dImmediateContext->CSSetShader(nullptr, nullptr, 0);

	ID3D11UnorderedAccessView* ppUAViewnullptr[1] = { nullptr };
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewnullptr, nullptr);

	ID3D11ShaderResourceView* ppSRVnullptr[2] = { nullptr, nullptr };
	pd3dImmediateContext->CSSetShaderResources(0, 2, ppSRVnullptr);

	ID3D11Buffer* ppCBnullptr[1] = { nullptr };
	pd3dImmediateContext->CSSetConstantBuffers(0, 1, ppCBnullptr);
}


ID3D11Buffer* DCSim::CreateAndCopyToDebugBuf(ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer)
{
	ID3D11Buffer* debugbuf = nullptr;
	
	D3D11_BUFFER_DESC desc;
	ZeroMemory(&desc, sizeof(desc));
	pBuffer->GetDesc(&desc);
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.MiscFlags = 0;

	

	if (SUCCEEDED(pDevice->CreateBuffer(&desc, nullptr, &debugbuf)))
	{
		pd3dImmediateContext->CopyResource(debugbuf, pBuffer);	
	}
	

	return debugbuf;
}