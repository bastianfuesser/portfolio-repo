cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

# Find VTK package
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Find Eigen package (assumed installed system-wide)
find_package(Eigen3 REQUIRED)

# Include Eigen headers
include_directories(${EIGEN3_INCLUDE_DIR})

# Define your sources and executable
set(SOURCES
    main_script.cpp
    functions.cpp
)

# Create a library for your project
add_library(MyProjectLib ${SOURCES})

# Link VTK libraries to the library
target_link_libraries(MyProjectLib ${VTK_LIBRARIES})

# Add an executable using the library
add_executable(simulation_program main_script.cpp)
target_link_libraries(simulation_program MyProjectLib)

# Installation setup (install binaries and headers if needed)
install(TARGETS simulation_program DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# install header
install(FILES function.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

