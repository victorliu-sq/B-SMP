#pragma once

#include <chrono>
#include <iostream>
#include <vector>
#include <glog/logging.h>

#include "types.h"
#include <signal.h>


// ===================================================================================
// host macros
#define SLEEP_MILLISECONDS(us) std::this_thread::sleep_for(std::chrono::milliseconds(us))

// Overflow Problem of index
#define IDX_MUL(a, b) (static_cast<size_t>(a) * static_cast<size_t>(b))
#define SIZE_MUL(a, b) (static_cast<size_t>(a) * static_cast<size_t>(b))
#define IDX_ADD(a, b) (static_cast<size_t>(a) + static_cast<size_t>(b))
#define IDX_MUL_ADD(a, b, c)  (static_cast<size_t>(a) * static_cast<size_t>(b) + static_cast<size_t>(c))

// ========================== Filesystem Operations ==============================
#define FileExists(x) std::filesystem::exists(x)
#define MakeDirec(x) std::filesystem::create_directories(x)
#define RemoveFile(x) std::filesystem::remove(x)           // Removes a single file
#define RemoveAll(x) std::filesystem::remove_all(x)        // Removes files and directories recursively

// ===================================================================================
// host static methods
// only visible to the this namespace
namespace bsmp {
  static uint64_t getNanoSecond() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  inline std::ostream& operator<<(std::ostream& os, const bsmp::Matching& matching) {
    std::ostringstream oss;
    oss << "Final Matching Results:\n";

    for (size_t m = 0; m < matching.size(); ++m) {
      oss << "  Man M" << m << " ↔ Woman W" << matching[m] << "\n";
    }

    os << oss.str();
    return os;
  }
}
