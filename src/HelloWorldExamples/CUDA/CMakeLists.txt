# ============================================================= #
# CUDA Vector Addition
# Adam Söderström, 2018-03-15, adaso578@student.liu.se
# ============================================================= #

# Project definition
cmake_minimum_required(VERSION 3.1)
project(CudaVectorAddition LANGUAGES CXX CUDA)


find_package(CUDA QUIET REQUIRED)

add_executable(${PROJECT_NAME}  	main.cu)


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
include_directories(${PROJECT_NAME} ${SRC_DIR}
									${CUDA_TOOLKIT_ROOT_DIR}/include)
