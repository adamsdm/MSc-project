# Project definition
cmake_minimum_required(VERSION 3.1)
project(OpenCLBarnesHut)



set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})


# OpenCL
find_package( OpenCL REQUIRED )
MESSAGE( STATUS "CL FOUND?: " ${OPENCL_FOUND} )
MESSAGE( STATUS "INC. DIRS: " ${OPENCL_INCLUDE_DIRS} )
MESSAGE( STATUS "INC. LIBS: " ${OPENCL_LIBRARIES} )
MESSAGE( STATUS "OCL Versi: " ${OPENCL_VERSION_STRING} )
MESSAGE( STATUS "HAS CXX??: " ${OPENCL_HAS_CXX} )



set(SRC_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/Source")
set(EXT_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/External")
set(KRL_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/CLKernels")
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
file(GLOB PROJECT_KERNELS   ${KRL_DIR}/*.cl)


source_group("Sources" FILES ${PROJECT_SOURCES})
source_group("Headers" FILES ${PROJECT_HEADERS})
source_group("Shaders" FILES ${PROJECT_SHADERS})
source_group("Kernels" FILES ${PROJECT_KERNELS})


add_executable(${PROJECT_NAME}  	${PROJECT_SOURCES}
                                	${PROJECT_HEADERS}
                                	${PROJECT_SHADERS}
                                    	${PROJECT_KERNELS})



set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
include_directories(${PROJECT_NAME} ${SRC_DIR}
                                    ${GLM_DIR}
                                    ${GLFW_DIR}/include
                                    ${GLAD_DIR}/include
                                    ${OPENCL_INCLUDE_DIR})


# OpenCL


target_link_libraries(${PROJECT_NAME} ${OPENCL_LIBRARIES})

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



