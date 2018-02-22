# OpenGL Boilerplate (wip)

A minimalistic OpenGL boilerplate based on [Learn OpenGL](https://learnopengl.com/) and [initGL](https://github.com/aaronmjacobs/InitGL).

Technologies used:
* CMake
* GLFW
* GLM
* GLAD

## Installation
The installation precedure requiers you to have [CMake](https://cmake.org/) installed.

Clone the repository with `git clone https://github.com/adamsdm/OpenGL-Boilerplate.git --recursive`



### Windows
* Install using CMAKE GUI
    * Set source location (the top folder where CMakeLists.txt are located)
    * Set build location, preferably into a seperate build folder to keep things nice and clean
    * Hit Configure
    * Select relevant generator from the dropdown list
    * Hit Generate
* Open up the ALL_BUILD solution from your build-folder and build it in Visual Studio
* Set OpenGL-Boilerplate as startup project (right click, select "Set as StartUp Project")

### OSX 
Not tested :exclamation:

### Linux 
Not tested :exclamation:

# TODO
* Copy resource files to binary folder in a better way
* Abstract main class