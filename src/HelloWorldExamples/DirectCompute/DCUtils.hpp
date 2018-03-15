#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } }
#endif

namespace DCUtils
{
	//--------------------------------------------------------------------------------------
	// Create the D3D device and device context suitable for running Compute Shaders(CS)
	//--------------------------------------------------------------------------------------
	HRESULT CreateComputeDevice(_Outptr_ ID3D11Device** ppDeviceOut, _Outptr_ ID3D11DeviceContext** ppContextOut, _In_ bool bForceRef);

	//--------------------------------------------------------------------------------------
	// Compile and create the CS
	//--------------------------------------------------------------------------------------
	HRESULT CreateComputeShader(_In_z_ LPCWSTR pSrcFile, _In_z_ LPCSTR pFunctionName,
		_In_ ID3D11Device* pDevice, _Outptr_ ID3D11ComputeShader** ppShaderOut);


	//--------------------------------------------------------------------------------------
	// Tries to find the location of the shader file
	// This is a trimmed down version of DXUTFindDXSDKMediaFileCch.
	//--------------------------------------------------------------------------------------
	HRESULT FindDXSDKShaderFileCch(_Out_writes_(cchDest) WCHAR* strDestPath, _In_ int cchDest, _In_z_ LPCWSTR strFilename);

	//--------------------------------------------------------------------------------------
	// Create Structured Buffer
	//--------------------------------------------------------------------------------------
	HRESULT CreateStructuredBuffer(_In_ ID3D11Device* pDevice, _In_ UINT uElementSize, _In_ UINT uCount,
		_In_reads_(uElementSize*uCount) void* pInitData,
		_Outptr_ ID3D11Buffer** ppBufOut);

	//--------------------------------------------------------------------------------------
	// Create Shader Resource View for Structured or Raw Buffers
	//--------------------------------------------------------------------------------------
	HRESULT CreateBufferSRV(_In_ ID3D11Device* pDevice, _In_ ID3D11Buffer* pBuffer, _Outptr_ ID3D11ShaderResourceView** ppSRVOut);

	//--------------------------------------------------------------------------------------
	// Create Unordered Access View for Structured or Raw Buffers
	//-------------------------------------------------------------------------------------- 
	HRESULT CreateBufferUAV(_In_ ID3D11Device* pDevice, _In_ ID3D11Buffer* pBuffer, _Outptr_ ID3D11UnorderedAccessView** pUAVOut);

	//--------------------------------------------------------------------------------------
	// Run CS
	//-------------------------------------------------------------------------------------- 
	void RunComputeShader(_In_ ID3D11DeviceContext* pd3dImmediateContext,
		_In_ ID3D11ComputeShader* pComputeShader,
		_In_ UINT nNumViews, _In_reads_(nNumViews) ID3D11ShaderResourceView** pShaderResourceViews,
		_In_opt_ ID3D11Buffer* pCBCS, _In_reads_opt_(dwNumDataBytes) void* pCSData, _In_ DWORD dwNumDataBytes,
		_In_ ID3D11UnorderedAccessView* pUnorderedAccessView,
		_In_ UINT X, _In_ UINT Y, _In_ UINT Z);

	//--------------------------------------------------------------------------------------
	// Create a CPU accessible buffer and download the content of a GPU buffer into it
	// This function is very useful for debugging CS programs
	//-------------------------------------------------------------------------------------- 
	ID3D11Buffer* CreateAndCopyToDebugBuf(ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer);

}