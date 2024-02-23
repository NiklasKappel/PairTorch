cmake_minimum_required(VERSION 3.16)

project(PairTorch VERSION 1.0 LANGUAGES CXX)

include(CheckIncludeFileCXX)  # Required by LAMMPSInterfacePlugin.
# include(LAMMPSInterfacePlugin)  # Load from external submodule.

add_subdirectory(src)
