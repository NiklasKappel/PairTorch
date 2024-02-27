cmake_minimum_required(VERSION 3.16)

project(PairTorch VERSION 1.0 LANGUAGES CXX)

set(DEFAULT_LAMMPS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/lammps/src")
set(LAMMPS_SOURCE_DIR "${DEFAULT_LAMMPS_SOURCE_DIR}" CACHE PATH "Location of LAMMPS sources folder")
get_filename_component(LAMMPS_DIR "${LAMMPS_SOURCE_DIR}" DIRECTORY)

if(LAMMPS_SOURCE_DIR STREQUAL DEFAULT_LAMMPS_SOURCE_DIR)
  # Ensure that submodule files are present.
  # See https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html.
  # See https://git-scm.com/book/en/v2/Git-Tools-Submodules.
  find_package(Git QUIET)
  if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
      message(STATUS "Submodule update")
      execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
                      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      RESULT_VARIABLE GIT_SUBMOD_RESULT)
      if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "Command `git submodule update --init --recursive` failed with ${GIT_SUBMOD_RESULT}. Please checkout submodules.")
      endif()
    endif()
  endif()

  if(NOT EXISTS "${LAMMPS_DIR}/README")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
  endif()
endif()

# See https://github.com/lammps/lammps/blob/release/examples/kim/plugin/CMakeLists.txt#L11.
include(CheckIncludeFileCXX)  # Required by LAMMPSInterfacePlugin.cmake.
include("${LAMMPS_DIR}/cmake/Modules/LAMMPSInterfacePlugin.cmake")

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
