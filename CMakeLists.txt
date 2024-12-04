cmake_minimum_required(VERSION 3.10)
project(export_model_test)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Add libtorch path
set(TORCH_DIR ~/rlio_ws/libtorch)

# Include libtorch
find_package(Torch REQUIRED PATHS ${TORCH_DIR}/share/cmake/Torch)

# Add executable
add_executable(export_model_test export_model_test.cpp)

# Link libtorch
target_link_libraries(export_model_test "${TORCH_LIBRARIES}")
set_property(TARGET export_model_test PROPERTY CXX_STANDARD 17)
