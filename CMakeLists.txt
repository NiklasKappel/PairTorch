cmake_minimum_required(VERSION 3.16)

project(PairTorch VERSION 1.0 LANGUAGES CXX)

# See https://github.com/lammps/lammps/blob/release/cmake/Modules/LAMMPSInterfacePlugin.cmake#L13.
set(LAMMPS_SOURCE_DIR "" CACHE PATH "Location of LAMMPS sources folder")
get_filename_component(LAMMPS_DIR "${LAMMPS_SOURCE_DIR}" DIRECTORY)

# See https://github.com/lammps/lammps/blob/release/examples/kim/plugin/CMakeLists.txt#L12.
include(CheckIncludeFileCXX)  # Required by LAMMPSInterfacePlugin.cmake.
include("${LAMMPS_DIR}/cmake/Modules/LAMMPSInterfacePlugin.cmake")

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
