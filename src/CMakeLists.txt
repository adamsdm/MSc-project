# ============================================================================= #
# Root CMakeLists for N-Body Barnes-Hut
# Adam Söderström, 2018-03-15, adaso578@student.liu.se
# ============================================================================= #
# Once configured these variables will be defined and passed to subdirectories:
#	EXT_DIR			- Folder containing external libraries
#	GLM_DIR			- Directory inside externals containing glm
#	GLFW_DIR		- Directory containing GLFW
#	GLAD_DIR		- Folder containing glad source
#	COMMON_DIR		- Folder containing common source files
#	COMMON_SOURCES	- Source files in common dir
#
#	PROJECT_SHADERS	- *.hlsl files stored in Shaders/
# 
# ============================================================================= #

cmake_minimum_required(VERSION 3.1)


set(COMMON_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/Common")
set(EXT_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/External")
set(GLM_DIR     "${EXT_DIR}/glm")
set(GLFW_DIR    "${EXT_DIR}/glfw")
set(GLAD_DIR    "${EXT_DIR}/glad")


file(GLOB PROJECT_SHADERS   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl)
file(GLOB COMMON_SOURCES	${COMMON_DIR}/*.h ${COMMON_DIR}/*.cpp)

# Microsoft Visual C++
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++11")
    if(NOT WIN32)
        set(GLAD_LIBRARIES dl)
    endif()
endif()


# setup CMake config file
configure_file(${COMMON_DIR}/config.h.in ${CMAKE_BINARY_DIR}/config.h)


include_directories(${GLFW_DIR}/include
                    ${GLAD_DIR}/include
					${GLM_DIR}
					${CMAKE_BINARY_DIR})


# GLFW
option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)
add_subdirectory("${GLFW_DIR}")


# GLAD
add_library("glad" "${GLAD_DIR}/src/glad.c")
target_link_libraries("glad" "${CMAKE_DL_LIBS}")


option(BUILD_ALL "Builds all framework implementations" OFF)
option(BUILD_SEQUENTIAL "Builds sequential implemenation" OFF)
option(BUILD_CUDA "Builds CUDA implemenation" OFF)
option(BUILD_OPENCL "Builds OpenCL implemenation" OFF)
option(BUILD_DIRECTCOMPUTE "Builds DirectCompute implemenation" OFF)
option(BUILD_SKEPU "Builds SkePU implemenation" OFF)


# Building options

option(BUILD_TESTING "Builds and enables testing" OFF)

if(BUILD_TESTING)
	message(WARNING "Building for tests")
	file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/CSV)
endif(BUILD_TESTING)

if(BUILD_ALL)
		add_subdirectory(Sequential) 
		add_subdirectory(CUDA) 
		add_subdirectory(OpenCL) 
		add_subdirectory(DirectCompute) 
		add_subdirectory(SkePU)
else(BUILD_ALL)

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

	if(BUILD_SKEPU)
		add_subdirectory(SkePU) 
	endif(BUILD_SKEPU)

endif(BUILD_ALL)

