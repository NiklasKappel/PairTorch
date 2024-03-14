add_library(PairTorch MODULE pair_torch.cpp torchplugin.cpp)
target_compile_features(PairTorch PRIVATE cxx_std_17)
target_link_libraries(PairTorch PRIVATE lammps)
target_link_libraries(PairTorch PRIVATE "${TORCH_LIBRARIES}")
set_target_properties(PairTorch PROPERTIES PREFIX "" SUFFIX ".so")

# See https://github.com/lammps/lammps/blob/release/examples/kim/plugin/CMakeLists.txt#L47.
# MacOS seems to need this.
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  set_target_properties(PairTorch PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # Tell CMake to export all symbols to a .dll on Windows with special case for MinGW cross-compilers.
  set_target_properties(PairTorch PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(PairTorch PROPERTIES LINK_FLAGS "-Wl,--export-all-symbols")
  endif()
else()
  set_target_properties(PairTorch PROPERTIES LINK_FLAGS "-rdynamic")
endif()

install(
  TARGETS PairTorch
  LIBRARY
  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
