cmake_minimum_required (VERSION 3.20)

# set(CMAKE_BUILD_TYPE Debug) # Do not set if profiling using NSight
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 86)

project (ece408project LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_library(ece408net ece408net.cc ece408net.h)

add_executable(m1_cpu m1_cpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/new-forward.cu")
target_link_libraries(m1_cpu ece408net MiniDNNLib)
# Set compiler flags for debug builds
target_compile_options(m1_cpu PRIVATE "$<$<CONFIG:DEBUG>:-gdwarf-5>")

add_executable(m1_gpu m1_gpu.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/new-forward.cu")
target_link_libraries(m1_gpu ece408net MiniDNNLib)

add_executable(m2 m2.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/matmul-unroll.cu")
target_link_libraries(m2 ece408net MiniDNNLib)

add_executable(m3 m3.cc "${PROJECT_SOURCE_DIR}/src/layer/custom/m3-forward.cu")
target_link_libraries(m3 ece408net MiniDNNLib cublas)