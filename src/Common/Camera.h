#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>


enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 100.0f;
const float SENSITIVTY = 0.15f;
const float ZOOM = 45.0f;

class Camera {
public:
	// Camera Attributes
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;
	// Eular Angles
	float Yaw;
	float Pitch;
	// Camera options
	float MovementSpeed;
	float MouseSensitivity;
	float Zoom;

	/**
	* Constructur with vectors
	* @param position	initial camera position
	* @param up			World up axis unit vector
	* @param yaw		Camera yaw
	* @param pitch		Camera pitch
	*/
	Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);

	/**
	* Constructur with scalars
	* @param position	initial camera position
	* @param up			World up axis unit vector
	* @param yaw		Camera yaw
	* @param pitch		Camera pitch
	*/
	Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);
	
	/**
	* Returns view matrix
	* @return <glm::mat4> View matrix
	*/
	glm::mat4 GetViewMatrix();

	/**
	* Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	* @param direction		Camera_Movement enum giving the direction
	* @param deltaTime	
	*/
	void ProcessKeyboard(Camera_Movement direction, float deltaTime);

	
	/**
	* Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	* @param xoffset		delta x
	* @param yoffset		delta y
	* @param constrainPitch
	*/
	void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);

	/**
	* Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
	* @param yoffset		delta y
	*/
	void ProcessMouseScroll(float yoffset);

private:
	// Calculates the front vector from the Camera's (updated) Eular Angles
	void updateCameraVectors();
};