#ifndef B_SMP_TYPES_H
#define B_SMP_TYPES_H
#include <thrust/mr/allocator.h>
#include "common/types.h"

namespace bsmp {
  // Host-side pinned vector (accelerates H2D transfer)
  using MrPinned = thrust::system::cuda::universal_host_pinned_memory_resource;

  template <typename T>
  using HVector = thrust::host_vector<T, thrust::mr::stateless_resource_allocator<T, MrPinned>>;

  // template<typename T>
  // using HVector = thrust::host_vector<T>; // Normal host vector, NOT pinned memory

  template <typename T>
  class HArray : public HVector<T> {
  public:
    // Import all constructors from HVector<T> into this derived class.
    using HVector<T>::HVector;

    T* GetRawPtr() {
      return thrust::raw_pointer_cast(this->data());
    }
  };

  // Device-side CUDA vector
  template <typename T>
  using DVector = thrust::device_vector<T>;
}

#endif //B_SMP_TYPES_H
