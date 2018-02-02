#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 particlePos;

out vec2 TexCoord;
out vec3 vParticlePos;
out vec3 vColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


void main()
{
	
	// Source: http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/
	vec3 CameraRight_worldspace = vec3(view[0][0], view[1][0], view[2][0]);
	vec3 CameraUp_worldspace = vec3(view[0][1], view[1][1], view[2][1]);
	vec3 particleCenter_wordspace = particlePos.xyz;
	
	vec3 vertexPosition_worldspace = 
		particleCenter_wordspace
		+ CameraRight_worldspace * aPos.x * 1.0f
		+ CameraUp_worldspace * aPos.y * 1.0f;


    gl_Position = projection * view * model * vec4(vertexPosition_worldspace, 1.0);

    TexCoord = aPos.xy + vec2(0.5, 0.5);
	vParticlePos = particlePos;
}