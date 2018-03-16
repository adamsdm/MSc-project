# Hardware accelerated N-Body Barnes-Hut simulation
This repository contains my M. Sc. work in in Media Technology and Engineering.
A visualization of the N-Body problem is implemented in various GPGPU frameworks when optimized using the hierarchical tree-based Barnes-Hut force calculation algorithm. The aim of this project is to evaluate GPGPU frameworks in terms of portability, code complexity and framework features. The selected framworks that is evaluated are: 
* CUDA
* OpenCL
* Microsoft DirectX Compute Shaders, aka DirectCompute.
* SkePU 2

![Particle system](https://raw.githubusercontent.com/adamsdm/MSc-project/master/report/Method/Figs/PSNoBounds.png "Particle system")
![Particle system with bounds](https://raw.githubusercontent.com/adamsdm/MSc-project/master/report/Method/Figs/PSWithBounds.png "Particle system")


## Table of contents
1. **[N-Body](#n-body)**
2. **[Barnes-Hut](#barnes-hut)**
3. **[Frameworks](#frameworks)**
4. **[Implementation](#implementation)**
5. **[Installation](#installation)**


## N-Body
An N-body simulation is a numerical approximation of the behaviour of bodies in a system. A common implementation of the N-Body problem is a astrophysical simulation where each body represents a celestial body such as a star, galaxy or planet. Other astrophysical applications of the N-body problem can be used on a smaller scale to simulate a e.g. 3-body simulation of the earth, moon and sun. The simulation approximates how the celestial bodies behave over time when each body is affected by gravitational forces from all the others. It has also been used in physical cosmology, where N-Body simulations have been used to study the formation of e.g. galaxy filaments and galaxy halos from the influence of dark matter. The simulation made in this work is a astrophysical simulation of a cluster of stars, where each star is affected by gravitational forces from all others.

## Barnes-Hut
Invented by J. Barnes and P. Hut, the Barnes-Hut algorithm is a hierarchical tree based force calculation algorithm with the time complexity O(n log n). The algorithm uses a hierarchical tree-structured subdivision of space into cubic cells, each divided into eight subcells whenever more than one particle is found to occupy the same cell. The root of the tree does thus represent the entire spatial space the particles reside in. When calculating the force applied to a single particle, only particles that are close enough under a certain condition, will be accurately force calculated. Particles that are far away from the particle will have a small impact on the resulting force, and can thus be approximated. In Barnes-Hut this is done by calculating each cells center of mass after the tree has been constructed. The tree is then traversed for each particle, if the cell's center of mass is far enough away from the particle, the entire subtree of that cell is approximated by a single "particle" at the cell's center of mass. If however the cell's center of mass is not far enough away from the particle the cell's subtree must be traversed \cite{singh1995load}. 

## Frameworks
### CUDA 
Released in 2007, CUDA developed by Nvidia was the first major GPGPU framework to be released. It aims to make the parallelization of a problem more manageable by providing an easy to work with API. One downside of CUDA is that is has a weak portability and can only be run on Nvidia GPU's. Despite this limitation it is a very popular framework among GPGPU developers. The installation procedure is very simple, all that is needed is a CUDA development toolkit which can be downloaded from Nvidia's webpage.

### OpenCL 
For a couple of years, CUDA was the only framework developed for the sole purpose of GPGPU and Nvidia had no competitors on the GPGPU front. That is until Apple took the initiative to develop a competitor, and backed by a lot of major companies, OpenCL was developed. OpenCL sought to offer a more portable and wide variety of parallel devices, and OpenCL offers the ability to run parallel implementations on other devices than just GPU's such as FPGA's and ARM devices. OpenCL is an open standard, and implementations are available from Apple, AMD, Intel, Nvidia and more. Because of this, the portability of OpenCL is good, and it can be run on most systems, provided a parallel hardware is present in the system. Since there are multiple implementations of OpenCL, the setup procedure differs, but OpenCL is usually provided by the manufacturers drivers.

### DirectCompute
Initially released with the DirectX 11 API, DirectCompute is Microsoft's technology for GPGPU, and unlike CUDA or OpenCL which relies on launching kernels, DirectCompute runs a compute shader as a part of the graphics pipeline. Although released with the DirectX 11 API, DirectCompute runs on GPUs that use either DirectX 10 or 11. Since DirectCompute is a part of the DirectX API, no addition additional setup is required, but DirectCompute can only be run on Windows PCs that have a supported DirectX version.

### SkePU
SkePU is a high-level skeleton programming framework originally developed by Johan Enmyren and Christoph W. Kessler at Linköping University. The framework aims to simplify the parallelization of a implementation by using skeleton functions such as map and reduce which are common parallel algorithms, which makes SkePU very different from the previously mentioned frameworks. When using SkePU, the developer specifies what backend the implementation should run. The currently available backends are sequential C, OpenMP, OpenCL, CUDA, and multi-GPU OpenCL and CUDA. The major revision SkePU 2 was designed by August Ernstsson, Lu Li and Christoph Kessler and is today maintained by August Ernstsson.

## Implementation
The N-Body Barnes-Hut algorithm is implemented in the framworks mentioned above. To be able to use all of the selected frameworks, make sure that the system your running supports the respective frameworks. 

The two part of the N-Body Barnes-Hut algorithm that is the most performance demanding is the force calculation. Each bodys applied net force is calculated by compararing the positions and masses of the other bodies in the system. This can be well parallelized and hardware accelerated since each particles net force can be calculated in parallel. The force calculation is performed in parallel on the GPU in all of the discussed frameworks. Once the force calculation has been completed, the bodies positions is updated using a first order Euler integration.


## Installation
The implementation uses the cross platform software CMake build tool to generate necessary build files. To install this project, CMake is thus requiered. 

1. Launch the CMake GUI application
2. Enter the source directory and build directory (preferably seperate the build from the source directory)
3. Hit configure and select your choice of generator
4. Once the configuration has been completed, select which frameworks you want to build from the options listed
5. Hit configure again
6. Generate
