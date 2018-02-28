#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>

#include "Particle.h"
#include "OctreeNode.h"
#include "sOctreeNode.h"

#pragma comment(lib,"d3d11.lib")

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } }
#endif

class DCSim {
	
public:
	DCSim();
	~DCSim();

	void step(Particle *ParticlesContainer, sOctreeNode *nodeContainer, GLfloat *g_particule_position_size_data, int count, unsigned int MAX_PARTICLES, float dt);

private:
	HRESULT CreateComputeDevice(ID3D11Device** ppDeviceOut, ID3D11DeviceContext** ppContextOut, bool bForceRef);

};