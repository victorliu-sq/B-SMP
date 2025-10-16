cmake_minimum_required(VERSION 3.19)

project(B-SMP
        LANGUAGES C CXX CUDA)

# ================== C++ Compiling Flags ======================
# set c++ standard
set(CMAKE_CXX_STANDARD 17)
# Threads flag for C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# Optimization flags for C++
set(CMAKE_CXX_FLAGS_RELEASE "-w -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -O0")

# ================== CUDA Compiling Flags ======================
# Auto-detect from currently visible GPUs
set(CMAKE_CUDA_ARCHITECTURES "native")
# Device Lambda flag for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# Optimization flags for CUDA
set(CMAKE_CUDA_FLAGS_RELEASE "-w -Wno-deprecated-gpu-targets -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -Wall -Wextra -O0")

# ================== Dependencies ======================
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party")

find_package(GTest 1.15.0 REQUIRED) # Notice the GT
find_package(gflags 2.2.2 REQUIRED)
find_package(glog 0.6.0 REQUIRED)
find_package(benchmark 1.9.4 REQUIRED)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ================== Subdirectories ======================
add_subdirectory(src)
add_subdirectory(example)
