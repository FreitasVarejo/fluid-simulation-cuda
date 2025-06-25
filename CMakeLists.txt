cmake_minimum_required(VERSION 3.20)
project(fluid-waves LANGUAGES C CXX CUDA)

# ───────── padrões de linguagem ──────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)

# ───────── opções editáveis (tunning via -D) ─────────────────────────────────
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

# ───────── flags globais de otimização ──────────────────────────────────────
# C / C++
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -march=native -ffast-math -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -Wall")

# CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3")

# ───────── OpenMP ───────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)          # falha cedo se não houver suporte
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#add_compile_definitions(_OPENMP)

# ───────── includes ─────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ───────── biblioteca comum ─────────────────────────────────────────────────
add_library(solver OBJECT src/solver.c)

# ───────── executáveis ──────────────────────────────────────────────────────
add_executable(serial  src/main_serial.c  $<TARGET_OBJECTS:solver>)
target_link_libraries(serial PRIVATE m)

add_executable(openmp  src/main_openmp.c  $<TARGET_OBJECTS:solver>)
target_link_libraries(openmp PRIVATE m OpenMP::OpenMP_C)

add_executable(cuda
    src/main_cuda.cu
    src/solver_cuda.cu
)
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda PRIVATE m)
