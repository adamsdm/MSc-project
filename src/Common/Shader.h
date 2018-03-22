#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader {

public:
	// ID of this shader instance
	unsigned int ID;

	/**
	* Creates a new shader instance with given source files
	* @param vertexPath		path to the vertex shader
	* @param fragmentPath	path to the fragment shader
	*/
	Shader(const GLchar* vertexPath, const GLchar* fragmentPath);
	
	/**
	* Set this shader as the active one
	*/
	void use();

	/**
	* Passes a uniform boolean to the shader
	* @param name		name of the uniform
	* @param value		boolean value
	*/
	void setBool(const std::string &name, bool value) const;

	/**
	* Passes a uniform integer to the shader
	* @param name		name of the uniform
	* @param value		integer value
	*/
	void setInt(const std::string &name, int value) const;	

	/**
	* Passes a uniform float to the shader
	* @param name		name of the uniform
	* @param value		float value
	*/
	void setFloat(const std::string &name, float value) const;

	/**
	* Passes a uniform glm::mat4 to the shader
	* @param name		name of the uniform
	* @param value		matrix
	*/
	void setMat4(const std::string &name,  glm::mat4 value) const;

private:
	void checkCompileErrors(unsigned int shader, std::string type);
};