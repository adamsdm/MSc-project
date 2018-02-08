#version 330 core
out vec4 FragColor;
  
in vec2 TexCoord;
in vec3 vParticlePos;
in vec3 vForce;

uniform sampler2D ourTexture;
uniform float time;

vec3 LerpRGB(vec3 a, vec3 b, float t)
{
	//return vec3( a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t);
	return vec3(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t);
}

void main()
{
	vec3 loFcolor = vec3(1.0f, 1.0f, 1.0f);
	vec3 hiFcolor = vec3(1.0f, 0.0f, 0.0f);

	// Calculate net force
	float fNet = sqrt(vForce.x*vForce.x + vForce.y*vForce.y + vForce.z*vForce.z);

	// Max force varies, but 800 is an avarage
	fNet = min(1.0f, fNet / 1000.0f);

	FragColor = texture(ourTexture, TexCoord) * vec4(1.0f, 1.0f, 1.0f, 0.3f);
	//FragColor = vec4(LerpRGB(loFcolor, hiFcolor, fNet), 0.5f);
	//FragColor = vec4(0.9f, 0.2f, 1.0f, 0.2f);
}