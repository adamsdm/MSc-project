// -------------------------------------------------------------------------------- //
// Based on Microsoft's Hello world example for DirectCompute						//
// https://msdn.microsoft.com/en-us/library/windows/desktop/ff476330(v=vs.85).aspx  //
// -------------------------------------------------------------------------------- //

#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>

#include "Particle.h"
#include "OctreeNode.h"
#include "sOctreeNode.h"

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } }
#endif

class DCSim {
	
public:
	DCSim();
	~DCSim();

	void step(Particle *ParticlesContainer, sOctreeNode *nodeContainer, GLfloat *g_particule_position_size_data, int count, unsigned int MAX_PARTICLES, float dt);
	void updPos(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt);

private:
	HRESULT CreateComputeDevice(ID3D11Device** ppDeviceOut, ID3D11DeviceContext** ppContextOut, bool bForceRef);
	HRESULT CreateComputeShader(LPCWSTR pSrcFile, LPCSTR pFunctionName, ID3D11Device* pDevice, ID3D11ComputeShader** ppShaderOut);

	HRESULT CreateStructuredBuffer(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, void* pInitData, ID3D11Buffer** ppBufOut);
	HRESULT CreateRawBuffer(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, void* pInitData, ID3D11Buffer** ppBufOut);

	HRESULT CreateBufferSRV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11ShaderResourceView** ppSRVOut); 
	HRESULT CreateBufferUAV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11UnorderedAccessView** pUAVOut);

	ID3D11Buffer* CreateAndCopyToDebugBuf(ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer);

	void RunComputeShader(ID3D11DeviceContext* pd3dImmediateContext, 
		ID3D11ComputeShader* pComputeShader,
		UINT nNumViews, 
		ID3D11ShaderResourceView** pShaderResourceViews,
		ID3D11Buffer* pCBCS, 
		void* pCSData, 
		DWORD dwNumDataBytes,
		ID3D11UnorderedAccessView* pUnorderedAccessView,
		UINT X, UINT Y, UINT Z);

	ID3D11Device* device = nullptr;
	ID3D11DeviceContext* context = nullptr;

	ID3D11ComputeShader* vecAddCS = nullptr;
	ID3D11ComputeShader* updPosCS = nullptr;
	ID3D11ComputeShader* updForCS = nullptr;

};