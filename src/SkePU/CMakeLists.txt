# ============================================================= #
# SkePU N-Body Barnes-Hut implementation
# Adam Söderström, 2018-03-15, adaso578@student.liu.se
# ============================================================= #

message(STATUS "Configuring OpenCL...")

# Project definition
cmake_minimum_required(VERSION 3.1)
project(SkePUNBodyBarnesHut)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})



# Update SRC_DIR variable to point to current source
set(SRC_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/Source")

file(GLOB PROJECT_SOURCES   ${SRC_DIR}/*.cpp)
file(GLOB PROJECT_HEADERS   ${SRC_DIR}/*.h)


source_group("Sources" FILES ${PROJECT_SOURCES})
source_group("Headers" FILES ${PROJECT_HEADERS})
source_group("Shaders" FILES ${PROJECT_SHADERS})
source_group("Common"  FILES ${COMMON_SOURCES})


add_executable(${PROJECT_NAME}  	${PROJECT_SOURCES}
                                	${PROJECT_HEADERS}
									${COMMON_SOURCES}
                                	${PROJECT_SHADERS})



set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
include_directories(${PROJECT_NAME} ${SRC_DIR}
									${COMMON_DIR}
									${SHADER_DIR}
                                    ${GLM_DIR}
                                    ${GLFW_DIR}/include
                                    ${GLAD_DIR}/include)


# OpenCL
target_link_libraries(${PROJECT_NAME} ${OPENCL_LIBRARIES})

# GLFW
target_link_libraries(${PROJECT_NAME} "glfw" "${GLFW_LIBRARIES}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "GLFW_INCLUDE_NONE")

# GLAD
target_link_libraries(${PROJECT_NAME} "glad" "${CMAKE_DL_LIBS}")



