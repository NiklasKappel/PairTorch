cmake_minimum_required(VERSION 3.17...3.24)
project(PairTorch VERSION 1.0 LANGUAGES CXX)

# See https://github.com/lammps/lammps/blob/release/cmake/Modules/LAMMPSInterfacePlugin.cmake#L13.
set(LAMMPS_SOURCE_DIR "" CACHE PATH "Location of LAMMPS sources folder")
if(NOT LAMMPS_SOURCE_DIR)
  message(FATAL_ERROR "Must set LAMMPS_SOURCE_DIR")
endif()

# See https://github.com/lammps/lammps/blob/release/examples/kim/plugin/CMakeLists.txt#L11.
include(CheckIncludeFileCXX)
include("${LAMMPS_SOURCE_DIR}/../cmake/Modules/LAMMPSInterfacePlugin.cmake")

if(APPLE)
  message(WARNING " LibTorch does not support CUDA on MacOS.\n Will build PairTorch for CPU usage only.")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.2.1.zip")
  else()
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.2.1.zip")
  endif()
else()
  find_package(CUDAToolkit)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.1)
    message(STATUS "Found CUDA version >= 12.1.")
    if(WIN32)
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.2.1%2Bcu121.zip")
    else()
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip")
    endif()
  elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
    message(STATUS "Found CUDA version >= 11.8.")
    if(WIN32)
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.2.1%2Bcu118.zip")
    else()
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu118.zip")
    endif()
  else()
    message(WARNING " Did not find a PyTorch compatible CUDA version (>= 11.8).\n Will build PairTorch for CPU usage only.")
    if(WIN32)
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.1%2Bcpu.zip")
    else()
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcpu.zip")
    endif()
  endif()
endif()

include(FetchContent)
FetchContent_Declare(Torch URL ${LIBTORCH_URL})
FetchContent_MakeAvailable(Torch)
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
find_package(Torch REQUIRED)

if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(GNUInstallDirs)

# See https://github.com/lammps/lammps/blob/release/cmake/CMakeLists.txt#L47.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (NOT CMAKE_CROSSCOMPILING))
    set(CMAKE_INSTALL_PREFIX "$ENV{USERPROFILE}/LAMMPS" CACHE PATH "Default install path" FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Default install path" FORCE)
  endif()
endif()

install(
  DIRECTORY ${torch_SOURCE_DIR}/lib/
  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
