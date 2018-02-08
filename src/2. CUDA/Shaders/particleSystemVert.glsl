#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 particlePos;
layout (location = 2) in vec3 particleForce;

out vec2 TexCoord;
out vec3 vParticlePos;
out vec3 vForce;

uniform mat4 view;
uniform mat4 projection;
uniform float time;


void main()
{
	

	// Source: http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/
	vec3 CameraRight_worldspace = vec3(view[0][0], view[1][0], view[2][0]);
	vec3 CameraUp_worldspace = vec3(view[0][1], view[1][1], view[2][1]);
	vec3 particleCenter_wordspace = particlePos.xyz;
	
	float particleSize = 1.0f;

	vec3 vertexPosition_worldspace = 
		particleCenter_wordspace
		+ CameraRight_worldspace * aPos.x * particleSize
		+ CameraUp_worldspace * aPos.y * particleSize;


	

    gl_Position = projection * view * vec4(vertexPosition_worldspace, 1.0);

    TexCoord = aPos.xy + vec2(0.5, 0.5);
	vParticlePos = particlePos;
	vForce = particleForce;
}