# ============================================================= #
# Root CMakeLists for N-Body Barnes-Hut
# Adam Söderström, 2018-03-15, adaso578@student.liu.se
# ============================================================= #
# Once built these variables will be defined and passed to subdirectories:
#	EXT_DIR			- Folder containing external libraries
#	GLM_DIR			- Directory inside externals containing glm
#	GLFW_DIR		- Directory containing GLFW
#	GLAD_DIR		- Folder containing glad source
#
#	PROJECT_SHADERS	- *.hlsl files stored in Shaders/
# 
# ============================================================= #

cmake_minimum_required(VERSION 3.1)


set(EXT_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/External")
set(GLM_DIR     "${EXT_DIR}/glm")
set(GLFW_DIR    "${EXT_DIR}/glfw")
set(GLAD_DIR    "${EXT_DIR}/glad")


file(GLOB PROJECT_SHADERS   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl)

# Microsoft Visual C++
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++11")
    if(NOT WIN32)
        set(GLAD_LIBRARIES dl)
    endif()
endif()


include_directories(${GLFW_DIR}/include
                    ${GLAD_DIR}/include
					${GLM_DIR})


# GLFW
option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)
add_subdirectory("${GLFW_DIR}")


# GLAD
add_library("glad" "${GLAD_DIR}/src/glad.c")
target_link_libraries("glad" "${CMAKE_DL_LIBS}")


option(BUILD_SEQUENTIAL "description" OFF)
option(BUILD_CUDA "description" OFF)
option(BUILD_OPENCL "description" OFF)
option(BUILD_DIRECTCOMPUTE "description" OFF)


# Building options
if(BUILD_SEQUENTIAL)
	add_subdirectory(Sequential) 
endif(BUILD_SEQUENTIAL)

if(BUILD_CUDA)
	add_subdirectory(CUDA) 
endif(BUILD_CUDA)

if(BUILD_OPENCL)
	add_subdirectory(OpenCL) 
endif(BUILD_OPENCL)

if(BUILD_DIRECTCOMPUTE)
	add_subdirectory(DirectCompute) 
endif(BUILD_DIRECTCOMPUTE)