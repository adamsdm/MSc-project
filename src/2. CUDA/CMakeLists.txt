# Project definition
cmake_minimum_required(VERSION 3.1)
project(CUDAOpenGL LANGUAGES CXX CUDA)


# CUDA
find_package(CUDA QUIET REQUIRED)


set(SRC_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/Source")
set(EXT_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/External")
set(GLM_DIR     "${EXT_DIR}/glm")
set(GLFW_DIR    "${EXT_DIR}/glfw")
set(GLAD_DIR    "${EXT_DIR}/glad")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++11")
    if(NOT WIN32)
        set(GLAD_LIBRARIES dl)
    endif()
endif()

file(GLOB PROJECT_SHADERS   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl)
file(GLOB PROJECT_SOURCES   ${SRC_DIR}/*.cpp)
file(GLOB PROJECT_HEADERS   ${SRC_DIR}/*.h)
file(GLOB CUDA_SOURCES	    ${SRC_DIR}/*.cu)
file(GLOB CUDA_HEADERS	    ${SRC_DIR}/*.cuh)

source_group("Sources" FILES ${PROJECT_SOURCES})
source_group("Headers" FILES ${PROJECT_HEADERS})
source_group("Shaders" FILES ${PROJECT_SHADERS})
source_group("CUDA"    FILES ${CUDA_SOURCES}
			     ${CUDA_HEADERS})


cuda_add_executable(${PROJECT_NAME}  	${PROJECT_SOURCES}
                                	${PROJECT_HEADERS}
                                	${PROJECT_SHADERS}
					${CUDA_SOURCES}
					${CUDA_HEADERS})






set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
include_directories(${PROJECT_NAME} ${SRC_DIR}
                                    ${GLM_DIR}
                                    ${GLFW_DIR}/include
                                    ${GLAD_DIR}/include
				    ${CUDA_TOOLKIT_ROOT_DIR}/include)


# GLFW
option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)
add_subdirectory("${GLFW_DIR}")

target_link_libraries(${PROJECT_NAME} "glfw" "${GLFW_LIBRARIES}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "GLFW_INCLUDE_NONE")

# GLAD
add_library("glad" "${GLAD_DIR}/src/glad.c")
target_link_libraries(${PROJECT_NAME} "glad" "${CMAKE_DL_LIBS}")

