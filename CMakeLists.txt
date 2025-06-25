cmake_minimum_required(VERSION 3.20)
project(fluid-waves LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)

# ---------- opções editáveis ----------
set(TILE     32 CACHE STRING "CPU cache-blocking tile")
set(TB        1 CACHE STRING "Temporal blocking factor")
set(BLOCK_X  32 CACHE STRING "CUDA blockDim.x")
set(BLOCK_Y   8 CACHE STRING "CUDA blockDim.y")

add_compile_definitions(
    TILE=${TILE}
    TB=${TB}
    BLOCK_X=${BLOCK_X}
    BLOCK_Y=${BLOCK_Y}
)

# ---------- includes & warnings ----------
include_directories(${CMAKE_SOURCE_DIR}/include)
add_compile_options(-Wall)

# ---------- OpenMP ----------
find_package(OpenMP)
if(OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# ---------- biblioteca comum ----------
add_library(solver OBJECT src/solver.c)

# ---------- executáveis ----------
add_executable(serial  src/main_serial.c  $<TARGET_OBJECTS:solver>)
target_link_libraries(serial PRIVATE m)

add_executable(openmp  src/main_openmp.c  $<TARGET_OBJECTS:solver>)
if(OpenMP_C_FOUND)
    target_link_libraries(openmp PRIVATE m OpenMP::OpenMP_C)
else()
    target_link_libraries(openmp PRIVATE m)
endif()

add_executable(cuda
    src/main_cuda.cu
    src/solver_cuda.cu   # versão otimizada
)
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(cuda PRIVATE m)
