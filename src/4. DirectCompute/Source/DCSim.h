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
#include "dxerr.h"

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } }
#endif
#ifndef CHECK_ERR
#define CHECK_ERR(x)    { hr = (x); if( FAILED(hr) ) { DXTraceW( __FILEW__, __LINE__, hr, L#x, true ); exit(1); } }
#endif
#ifndef CHECK_ERR
#define CHECK_ERR(x)	x;
#endif



class DCSim {
	
public:
	DCSim();
	~DCSim();

	/**
	* Calculates one simulation step without copying data back and forth more than necessary.
	* @param *p_container	array containing the particles
	* @param nodeContainer	array containing the nodes of the octree
	* @param *g_particule_position_size_data	OpenGL position buffer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param count	the number of nodes in the nodeContainer
	* @param dt	the stepsize, delta time
	*/
	void step(Particle *ParticlesContainer, sOctreeNode *nodeContainer, GLfloat *g_particule_position_size_data, int count, unsigned int MAX_PARTICLES, float dt);

	/**
	* Launches a kernel that updates the positions of the particles.
	* @param *ParticlesContainer	array containing the particles
	* @param *g_particule_position_size_data	OpenGL position buffer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param dt	the stepsize, delta time
	*/
	void updPos(Particle *ParticlesContainer, GLfloat *g_particule_position_size_data, unsigned int MAX_PARTICLES, float dt);

	/**
	* Launches a kernel that updates the forces of the particles.
	* @param *ParticlesContainer	array containing the particles
	* @param nodeContainer	array containing the nodes of the octree
	* @param count	the number of nodes in the nodeContainer
	* @param MAX_PARTICLES	the number of particles in the ParticlesContainer
	* @param dt	the stepsize, delta time
	*/
	void updFor(Particle *ParticlesContainer, sOctreeNode *nodeContainer, int count, int MAX_PARTICLES, float dt);

private:

	// Creates a compute device with feature level 11 or 10
	HRESULT CreateComputeDevice(ID3D11Device** ppDeviceOut, ID3D11DeviceContext** ppContextOut, bool bForceRef);

	// Creates and compiles a compute shader from source
	HRESULT CreateComputeShader(LPCWSTR pSrcFile, LPCSTR pFunctionName, ID3D11Device* pDevice, ID3D11ComputeShader** ppShaderOut);

	// Create buffers
	HRESULT CreateStructuredBuffer(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, void* pInitData, ID3D11Buffer** ppBufOut);
	HRESULT CreateRawBuffer(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, void* pInitData, ID3D11Buffer** ppBufOut);
	HRESULT CreateConstantBuffer(ID3D11Device* pDevice, UINT uElementSize, UINT uCount, void* pInitData, ID3D11Buffer** ppBufOut);

	// Crate shader resource views
	HRESULT CreateBufferSRV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11ShaderResourceView** ppSRVOut); 
	HRESULT CreateBufferUAV(ID3D11Device* pDevice, ID3D11Buffer* pBuffer, ID3D11UnorderedAccessView** pUAVOut);

	// Copies the content from a buffer to a debug buffer with CPU read access
	ID3D11Buffer* CreateAndCopyToDebugBuf(ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer);

	// Launches a compute shader
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