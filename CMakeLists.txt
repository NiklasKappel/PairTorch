cmake_minimum_required(VERSION 3.16)

project(PairTorch VERSION 1.0 LANGUAGES CXX)

# Ensure that submodule files are present.
# See https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html.
# See https://git-scm.com/book/en/v2/Git-Tools-Submodules.
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/lammps/README")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

include(CheckIncludeFileCXX)  # Required by LAMMPSInterfacePlugin.
# include(LAMMPSInterfacePlugin)  # Load from external submodule.

add_subdirectory(src)
