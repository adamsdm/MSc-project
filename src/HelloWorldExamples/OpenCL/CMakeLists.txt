# ============================================================= #
# OpenCL Vector Addition
# Adam Söderström, 2018-03-15, adaso578@student.liu.se
# ============================================================= #

# Project definition
cmake_minimum_required(VERSION 3.1)
project(OpenCLVectorAdd)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

# OpenCL
find_package( OpenCL REQUIRED )

add_executable(${PROJECT_NAME}  	main.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
include_directories(${PROJECT_NAME} ${OPENCL_INCLUDE_DIR})

# OpenCL
target_link_libraries(${PROJECT_NAME} ${OPENCL_LIBRARIES})



