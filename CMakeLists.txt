cmake_minimum_required(VERSION 3.16...3.24)
project(PairTorch VERSION 1.0 LANGUAGES CXX)

# See https://github.com/lammps/lammps/blob/release/cmake/Modules/LAMMPSInterfacePlugin.cmake#L13.
set(LAMMPS_SOURCE_DIR "" CACHE PATH "Location of LAMMPS sources folder")
if(NOT LAMMPS_SOURCE_DIR)
  message(FATAL_ERROR "Must set LAMMPS_SOURCE_DIR")
endif()

# See https://github.com/lammps/lammps/blob/release/examples/kim/plugin/CMakeLists.txt#L11.
include(CheckIncludeFileCXX)
include("${LAMMPS_SOURCE_DIR}/../cmake/Modules/LAMMPSInterfacePlugin.cmake")

include(FetchContent)
# TODO: Fetch GPU version of libtorch if a GPU is available.
FetchContent_Declare(
  Torch
  URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcpu.zip
)
FetchContent_MakeAvailable(Torch)
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
find_package(Torch REQUIRED)

if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
