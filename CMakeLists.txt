cmake_minimum_required(VERSION 3.16)

project(PairTorch VERSION 1.0 LANGUAGES CXX)

# See https://github.com/lammps/lammps/blob/release/cmake/Modules/LAMMPSInterfacePlugin.cmake#L13.
set(LAMMPS_SOURCE_DIR "" CACHE PATH "Location of LAMMPS sources folder")
if(NOT LAMMPS_SOURCE_DIR)
  message(FATAL_ERROR "Must set LAMMPS_SOURCE_DIR")
endif()

# See https://github.com/lammps/lammps/blob/release/examples/kim/plugin/CMakeLists.txt#L11.
include(CheckIncludeFileCXX)
include("${LAMMPS_SOURCE_DIR}/../cmake/Modules/LAMMPSInterfacePlugin.cmake")

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
