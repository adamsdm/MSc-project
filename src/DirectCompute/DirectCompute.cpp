
// ---------------------------------------------------------------------------
// Basic DirectX 11 Compute Shader 
// Based on Microsofts example BasicCompute11
// https://github.com/walbourn/directx-sdk-samples/tree/master/BasicCompute11
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>

#include "DCUtils.hpp"

#define NUM_ELEMENTS	1024


#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } }
#endif


struct BufType
{
	int i;
	float f;
};

int main()
{

	ID3D11Device*               g_pDevice = nullptr;
	ID3D11DeviceContext*        g_pContext = nullptr;
	ID3D11ComputeShader*        g_pCS = nullptr;

	ID3D11Buffer*               g_pBuf0 = nullptr;
	ID3D11Buffer*               g_pBuf1 = nullptr;
	ID3D11Buffer*               g_pBufResult = nullptr;

	ID3D11ShaderResourceView*   g_pBuf0SRV = nullptr;
	ID3D11ShaderResourceView*   g_pBuf1SRV = nullptr;
	ID3D11UnorderedAccessView*  g_pBufResultUAV = nullptr;

	// Create a device that the kernel will run on
	printf("Creating device...");
	if (FAILED(DCUtils::CreateComputeDevice(&g_pDevice, &g_pContext, false))) return 1;
	printf(" Done!\n");

	// Compile CS from source
	printf("Creating Compute Shader...");
	if (FAILED(DCUtils::CreateComputeShader(L"VectorAddCS.hlsl", "CSMain", g_pDevice, &g_pCS))) return 1;
	printf(" Done!\n");


	// Create input data and buffers
	printf("Creating buffers and filling them with initial data...");
	BufType g_vBuf0[NUM_ELEMENTS];
	BufType g_vBuf1[NUM_ELEMENTS];
	for (int i = 0; i < NUM_ELEMENTS; ++i)
	{
		g_vBuf0[i].i = i;
		g_vBuf0[i].f = (float)i;

		g_vBuf1[i].i = 2*i;
		g_vBuf1[i].f = 2.0f*(float)i;

	}
	DCUtils::CreateStructuredBuffer(g_pDevice, sizeof(BufType), NUM_ELEMENTS, &g_vBuf0[0], &g_pBuf0);
	DCUtils::CreateStructuredBuffer(g_pDevice, sizeof(BufType), NUM_ELEMENTS, &g_vBuf1[0], &g_pBuf1);
	DCUtils::CreateStructuredBuffer(g_pDevice, sizeof(BufType), NUM_ELEMENTS, nullptr, &g_pBufResult);


	// Create buffer views
	printf("Creating buffer views...");
	DCUtils::CreateBufferSRV(g_pDevice, g_pBuf0, &g_pBuf0SRV);
	DCUtils::CreateBufferSRV(g_pDevice, g_pBuf1, &g_pBuf1SRV);
	DCUtils::CreateBufferUAV(g_pDevice, g_pBufResult, &g_pBufResultUAV);
	printf("done\n");


	printf("Running Compute Shader...");
	ID3D11ShaderResourceView* aRViews[2] = { g_pBuf0SRV, g_pBuf1SRV };
	DCUtils::RunComputeShader(g_pContext, g_pCS, 2, aRViews, nullptr, nullptr, 0, g_pBufResultUAV, NUM_ELEMENTS, 1, 1);
	printf("done\n");


	// Read back the result from GPU, verify its correctness against result computed by CPU
	{
		ID3D11Buffer* debugbuf = DCUtils::CreateAndCopyToDebugBuf(g_pDevice, g_pContext, g_pBufResult);
		D3D11_MAPPED_SUBRESOURCE MappedResource;
		BufType *p;
		g_pContext->Map(debugbuf, 0, D3D11_MAP_READ, 0, &MappedResource);

		// Set a break point here and put down the expression "p, 1024" in your watch window to see what has been written out by our CS
		// This is also a common trick to debug CS programs.
		p = (BufType*)MappedResource.pData;

		// Verify that if Compute Shader has done right
		printf("Verifying against CPU result...");
		bool bSuccess = true;

		//Print first 10 results
		for (size_t i = 0; i < 10; i++) {
			printf("%d + %d = %d \n", g_vBuf0[i].i, g_vBuf1[i].i, p[i].i);
		}

		for (int i = 0; i < NUM_ELEMENTS; ++i)
			if ((p[i].i != g_vBuf0[i].i + g_vBuf1[i].i)
				|| (p[i].f != g_vBuf0[i].f + g_vBuf1[i].f)
				)
			{
				printf("failure\n");
				bSuccess = false;

				break;
			}
		if (bSuccess)
			printf("succeeded\n");

		g_pContext->Unmap(debugbuf, 0);

		SAFE_RELEASE(debugbuf);
	}

	printf("Cleaning up...\n");
	SAFE_RELEASE(g_pBuf0SRV);
	SAFE_RELEASE(g_pBuf1SRV);
	SAFE_RELEASE(g_pBufResultUAV);
	SAFE_RELEASE(g_pBuf0);
	SAFE_RELEASE(g_pBuf1);
	SAFE_RELEASE(g_pBufResult);
	SAFE_RELEASE(g_pCS);
	SAFE_RELEASE(g_pContext);
	SAFE_RELEASE(g_pDevice);


    return 0;
}

