#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Shader.h"
#include "Camera.h"
#include "Application.h"
#include "ParticleSystem.h"
#include "OctreeNode.h"


#define NO_PARTICLES	1024


// Global application state

bool render_bounds = false;

// Window dimensions
unsigned int W = 1200;
unsigned int H = 800;


// ********************************* //
// ************ Camera ************* //
// ********************************* //

Camera camera(glm::vec3(0.0f, 0.0f, 600.0f));

float lastX = W / 2.0f;
float lastY = H / 2.0f;
bool firstMouse = true;



float deltaTime = 0.0f;

// GLFW input callbacks
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);


int main() {


	// Create a window
	GLFWwindow* window = App::initialize(W, H);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, mouse_callback);

	// ********************************* //
	// ********* Setup Shaders ********* //
	// ********************************* //

	Shader particleShader("../Shaders/particleSystemVert.glsl", "../Shaders/particleSystemFrag.glsl");
	Shader boxShader("../Shaders/boxVert.glsl", "../Shaders/boxFrag.glsl");

	// ********************************* //
	// ********* Setup Textures ******** //
	// ********************************* //
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	// set the texture wrapping/filtering options (on the currently bound texture object)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// load and generate the texture
	int width, height, nrChannels;
	unsigned char *data = stbi_load("../Resources/sun.png", &width, &height, &nrChannels, 0);

	

	if (data)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);



	// ********************************** //
	// ********* Setup Particles ******** //
	// ********************************** //

	ParticleSystem particlesystem(NO_PARTICLES);
	
	

	// ********************************* //
	// *********** Main Loop *********** //
	// ********************************* //
	
	float time = glfwGetTime();
	float lastTime = glfwGetTime();
	int frameCount = 0;

	// Matrices
	glm::mat4 view;
	glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)W / H, 0.1f, 4000.0f);





	GLuint BoxVBO, BoxVAO, BoxEBO;
	glGenVertexArrays(1, &BoxVAO);
	glGenBuffers(1, &BoxVBO);
	glGenBuffers(1, &BoxEBO);





	while (!glfwWindowShouldClose(window))
	{
		// Update time
		deltaTime = glfwGetTime() - time;
		frameCount++;

		// FPS Counter
		double currentTime = glfwGetTime();
		frameCount++;
		if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
											 // printf and reset timer
			//printf("%f ms/frame\n", 1000.0 / double(frameCount));
			printf("%f fps\n", 1.0f/deltaTime);
			frameCount = 0;
			lastTime += 1.0;
		}
		
		time = glfwGetTime();

		

		// Check for user inputs
		processInput(window);

		// render clear colour
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Start drawing
		particleShader.use();
		
		
		// Update view matrix
		view = camera.GetViewMatrix();
		projection = glm::perspective(glm::radians(camera.Zoom), (float) W / H, 0.1f, 4000.0f);

		// Set uniforms
		particleShader.setMat4("view", view);
		particleShader.setMat4("projection", projection);
		particleShader.setFloat("time", time);
		particleShader.setFloat("deltaTime", deltaTime);

		
		// Render particle system
		particlesystem.render(deltaTime);
		

		if (render_bounds){

			boxShader.use();

			boxShader.setMat4("view", view);
			boxShader.setMat4("projection", projection);
			particlesystem.renderBounds(boxShader);
			
		}
				
		// IMPORTANT: Dealocate the tree, otherwise we get a huge memoryleak since new nodes are added recursively, resulting in an infinite tree
		particlesystem.getTree()->free();


		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Clean up
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	W = width;
	H = height;
	glViewport(0, 0, width, height);
}
void processInput(GLFWwindow* window) {


	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);


	float cameraSpeed = 5.0f * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		camera.ProcessKeyboard(UP, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		camera.ProcessKeyboard(DOWN, deltaTime);


	// State management
	if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
		render_bounds = true;
	if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS)
		render_bounds = false;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}