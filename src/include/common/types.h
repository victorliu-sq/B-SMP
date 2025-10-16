#ifndef TYPES_H
#define TYPES_H

#include <queue>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>
#include <filesystem>
#include <fstream>
#include <sstream>

// Standard vector (host-side standard memory)

namespace bsmp {
  template <typename T>
  using Vector = std::vector<T>;

  // using PreferenceLists = Vector<Vector<int>>;
  using PreferenceLists = Vector<int>;
  using RankMatrix = Vector<int>;

  // using FlatPreferenceList = Vector<int>; // length = n * n

  template <typename T>
  using UPtr = std::unique_ptr<T>;

  template <typename T>
  using SPtr = std::shared_ptr<T>;

  template <typename T>
  using Queue = std::queue<T>;

  using Matching = std::vector<int>;

#define Move(x) std::move(x)

  template <typename T, typename... Args>
  static inline auto MakeUPtr(Args&&... args) -> UPtr<T> {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static inline auto MakeSPtr(Args&&... args) -> SPtr<T> {
    return std::make_shared<T>(std::forward<Args>(args)...);
  }

  struct PRNode {
    int idx_;
    int rank_;
  };

  using index_t = uint64_t;
  using ns_t = uint64_t;
  using ms_t = double;

  using PRMatrix = std::vector<std::vector<PRNode>>;

  using String = std::string;

  // ========================== Stream Aliases ==============================
  using OutStringStream = std::ostringstream;
  using InStringStream = std::istringstream;

  using OutFStream = std::ofstream;
  using InFStream = std::ifstream;


  // =========================== SMP Workload ======================================
  enum WorkloadType {
    PERFECT,
    RANDOM,
    CONGESTED,
    SOLO
  };

  inline String WorkloadTypeToString(WorkloadType type) {
    switch (type) {
    case PERFECT: return "Perfect";
    case RANDOM: return "Random";
    case SOLO: return "Solo";
    case CONGESTED: return "Congested";
    default: return "Unknown";
    }
  }
}

#endif //TYPES_H
