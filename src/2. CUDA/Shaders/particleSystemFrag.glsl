#version 330 core
out vec4 FragColor;
  
in vec2 TexCoord;
in vec3 vParticlePos;
in vec3 vColor;

uniform sampler2D ourTexture;
uniform float time;

void main()
{
	vec3 blue = vec3(0.0f, 0.0f, 1.0f);
	vec3 white = vec3(1.0f, 1.0f, 1.0f);

	//FragColor = texture(ourTexture, TexCoord) * vec4(1.0f, 1.0f, 1.0f, 0.3f);
	FragColor = vec4(vColor, 0.3f);
}