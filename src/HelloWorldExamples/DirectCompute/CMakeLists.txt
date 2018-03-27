# ============================================================= #
# DirectCompute Vector Addition
# Adam Söderström, 2018-03-15, adaso578@student.liu.se
# ============================================================= #

# Project definition
cmake_minimum_required(VERSION 3.1)
project(DirectComputeVecAdd)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

file(GLOB COMPUTE_SHADERS   VectorAddCS.hlsl)
set_source_files_properties(${COMPUTE_SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

add_executable(${PROJECT_NAME}  	main.cpp
									dxerr.h
									dxerr.cpp
									${COMPUTE_SHADERS})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)



