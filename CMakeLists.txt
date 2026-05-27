cmake_minimum_required(VERSION 3.20)
project(path_tracing_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(path_tracing_cuda
    src/main.cu
    src/kernel.cu
)

target_include_directories(path_tracing_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/external
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(path_tracing_cuda CUDA::cudart)

set_target_properties(path_tracing_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)