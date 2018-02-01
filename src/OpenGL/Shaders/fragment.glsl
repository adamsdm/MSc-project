#version 330 core
out vec4 FragColor;
  
in vec2 TexCoord;
in vec3 vParticlePos;
in vec3 vColor;

uniform sampler2D ourTexture;
uniform float time;

void main()
{
    FragColor = vec4(vColor, 1.0f);  //texture(ourTexture, TexCoord);  
}