add_library(PairTorch MODULE pair_torch.cpp)
# target_link_libraries(PairTorch PRIVATE lammps)  # Uncomment when using LAMMPSInterfacePlugin.
set_target_properties(PairTorch PROPERTIES PREFIX "" SUFFIX ".so")

# MacOS seems to need this
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set_target_properties(PairTorch PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
# tell CMake to export all symbols to a .dll on Windows with special case for MinGW cross-compilers
  set_target_properties(PairTorch PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(PairTorch PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
  endif()
else()
  set_target_properties(PairTorch PROPERTIES LINK_FLAGS "-rdynamic")
endif()
