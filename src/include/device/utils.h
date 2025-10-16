#ifndef B_SMP_UTILS_H
#define B_SMP_UTILS_H
#include <iostream>
#include <ostream>

#include "types.h"

// ===================================================================================
// device macros

#if defined(__CUDACC__) || defined(__CUDABE__)
#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#define CONST_STATIC_INIT(...)
#else
#define DEV_HOST
#define DEV_HOST_INLINE
#define DEV_INLINE
#define CONST_STATIC_INIT(...) = __VA_ARGS__
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define MAX_GRID_SIZE (756)
#define MAX_BLOCK_SIZE (1024)
// #define MAX_BLOCK_SIZE (256)
// #define MAX_BLOCK_SIZE (64)

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)

#define PARTITION_SIZE(n, num_gpus, gpu_id) ((n) / (num_gpus) + ((gpu_id) < ((n) % (num_gpus)) ? 1 : 0))
#define PARTITION_START(n, num_gpus, gpu_id) (((n) / (num_gpus)) * (gpu_id) + min((gpu_id), (n) % (num_gpus)))

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__                                 \
                << " with err msg: " << cudaGetErrorString(rt) << std::endl;   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

#define CUDA_CHECK_CONTINUE(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__                                 \
                << " with err msg: " << cudaGetErrorString(rt) << std::endl;   \
    }                                                                          \
  } while (0);

#define SYNC_ALL_STREAMS(streams) \
for (auto &stream : (streams)) { \
stream.Sync(); \
}

#define FOR_EACH_GPU(num_gpus, gpu_id) \
for (int gpu_id = 0; gpu_id < (num_gpus); ++gpu_id) if ((cudaSetDevice(gpu_id), true))


// Raw pointer extraction macro
#define RawPtr(v) thrust::raw_pointer_cast((v).data())


// ===================================================================================
// device static methods
namespace bsmp {
  static DEV_HOST_INLINE size_t round_up(size_t numerator,
                                         size_t denominator) {
    return (numerator + denominator - 1) / denominator;
  }

  inline void KernelSizing(int& block_num, int& block_size, size_t work_size) {
    block_size = MAX_BLOCK_SIZE;
    block_num = std::min(MAX_GRID_SIZE, (int)round_up(work_size, block_size));
  }

  inline void KernelSizingMax(int& grid_size, int& block_size, size_t work_size) {
    int device;
    cudaDeviceProp device_prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));

    // Get maximum threads per block supported by the GPU
    // int max_threads_per_block = device_prop.maxThreadsPerBlock;
    int max_threads_per_block = 256;

    // Get maximum grid dimensions
    int max_grid_dim_x = device_prop.maxGridSize[0];

    // Set block size to the maximum supported or work size if smaller
    block_size = (work_size < max_threads_per_block) ? work_size : max_threads_per_block;

    // Calculate grid size based on block size
    grid_size = (work_size + block_size - 1) / block_size;

    // Cap grid size at the GPU's max grid size
    if (grid_size > max_grid_dim_x) {
      grid_size = max_grid_dim_x;
    }
  }

  template <typename T>
  inline void SafeCudaMemcpyAsync(DVector<T>& dst, const T* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream,
                                  const char* error_context) {
    cudaError_t status = cudaMemcpyAsync(RawPtr(dst), src, count * sizeof(T), kind, stream);
    CHECK_EQ(status, cudaSuccess) << "cudaMemcpyAsync (" << error_context << ") error: "
      << cudaGetErrorString(status);
  }
}

#endif //B_SMP_UTILS_H
