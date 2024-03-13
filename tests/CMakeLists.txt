add_executable(scratch scratch.cpp)
target_compile_features(scratch PRIVATE cxx_std_17)
target_link_libraries(scratch PRIVATE "${TORCH_LIBRARIES}")

add_test(NAME load_plugin COMMAND lmp -in "${CMAKE_CURRENT_SOURCE_DIR}/in.load_plugin")
add_test(NAME load_model COMMAND lmp -in "${CMAKE_CURRENT_SOURCE_DIR}/in.load_model")
