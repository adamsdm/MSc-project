#version 330 core
out vec4 FragColor;
  
in vec2 TexCoord;
in vec3 vParticlePos;

uniform sampler2D ourTexture;
uniform float time;

void main()
{
	FragColor = texture(ourTexture, TexCoord);  
}