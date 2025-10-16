#pragma once
#include <thread>
#include <random>

#include "common/stopwatch.h"
#include "common/types.h"
#include "device/dev_array.h"

namespace bsmp {
  struct SmpWorkload {
    // ====== Data Members ======
    Vector<int> pref_lists_m;
    Vector<int> pref_lists_w;
    int n = 0;
    WorkloadType type;
    int group_size = 5;

    // ====== Constructor ======
    explicit SmpWorkload(int n_, WorkloadType type_,
                         bool use_cache = true, int group_size_ = 5)
      : pref_lists_m(n_ * n_), pref_lists_w(n_ * n_),
        n(n_), type(type_), group_size(group_size_) {
      if (use_cache) {
        GenerateWorkloadCached();
      }
      else {
        GenerateWorkload();
      }
    }

    // ====== Move Constructor ======
    SmpWorkload(SmpWorkload&& other) noexcept
      : pref_lists_m(std::move(other.pref_lists_m)),
        pref_lists_w(std::move(other.pref_lists_w)),
        n(other.n),
        type(other.type),
        group_size(other.group_size) {
      // Leave 'other' in a valid but empty state
      other.n = 0;
      other.group_size = 0;
    }

    // ====== Move Assignment Operator ======
    SmpWorkload& operator=(SmpWorkload&& other) noexcept {
      if (this != &other) {
        pref_lists_m = std::move(other.pref_lists_m);
        pref_lists_w = std::move(other.pref_lists_w);
        n = other.n;
        type = other.type;
        group_size = other.group_size;

        // Reset the source
        other.n = 0;
        other.group_size = 0;
      }
      return *this;
    }

  private:
    // =================== WORKLOAD GENERATION =====================
    void GenerateWorkload() {
      StopWatch sw;
      switch (type) {
      case RANDOM:
        Impl::GeneratePrefListsRandom(pref_lists_m, n, group_size);
        Impl::GeneratePrefListsRandom(pref_lists_w, n, group_size);
        break;
      case CONGESTED:
        Impl::GeneratePrefListsCongested(pref_lists_m, n);
        Impl::GeneratePrefListsCongested(pref_lists_w, n);
        break;
      case SOLO:
        Impl::GeneratePrefListsManSolo(pref_lists_m, n);
        Impl::GeneratePrefListsWomanSolo(pref_lists_w, n);
        break;
      case PERFECT:
        Impl::GeneratePrefListsPerfect(pref_lists_m, n);
        Impl::GeneratePrefListsPerfect(pref_lists_w, n);
        break;
      default:
        LOG(ERROR) << "Unknown workload type";
        break;
      }
      sw.Stop();
      LOG(INFO) << "Generated workload type: " << WorkloadTypeToString(type)
        << ", size: " << n
        << ", elapsed time: " << sw.GetEclapsedMs() << " ms.";
    }

    void GenerateWorkloadCached() {
      MakeDirec(Impl::workloadDir);
      String filename = Impl::GetWorkloadFilename(type, n, group_size);

      if (FileExists(Impl::workloadDir + filename)) {
        LOG(INFO) << "Loaded existing dataset from " << filename;
        int n_file = 0;
        Impl::LoadSmpWorkload(filename, pref_lists_m, pref_lists_w, n_file);
        if (n_file != n) {
          LOG(WARNING) << "Loaded n(" << n_file << ") != requested n(" << n
            << "), using loaded size.";
          this->n = n_file;
        }
      }
      else {
        LOG(INFO) << "Dataset not found, generating new one.";
        GenerateWorkload();
        Impl::SaveSmpWorkload(filename, pref_lists_m, pref_lists_w, n);
        LOG(INFO) << "Saved generated dataset to " << filename;
      }
    }

    // =================== IMPLEMENTATION CORE =====================
    class Impl {
    public:
      static constexpr const char* workloadDir = "data/workloads/";

      // ============================================================
      // ======================= CONGESTED ==========================
      static void GeneratePrefListsCongested(PreferenceLists& pl, int n) {
        Vector<int> row(n);
        for (int i = 0; i < n; ++i) row[i] = i;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(row.begin(), row.end(), gen);

        const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
        const int rows_per_thread = (n + (int)maxThreads - 1) / (int)maxThreads;

        std::vector<std::thread> threads;
        threads.reserve(maxThreads);

        auto worker = [&](int r0, int r1) {
          for (int r = r0; r < r1; ++r) {
            int base = r * n;
            for (int c = 0; c < n; ++c)
              pl[base + c] = row[c];
          }
        };

        for (unsigned t = 0; t < maxThreads; ++t) {
          int r0 = (int)t * rows_per_thread;
          int r1 = std::min(r0 + rows_per_thread, n);
          if (r0 >= r1) break;
          threads.emplace_back(worker, r0, r1);
        }
        for (auto& th : threads) th.join();
      }

      // ======================= PERFECT ============================
      static void GeneratePrefListsPerfect(PreferenceLists& pl, int n) {
        Vector<int> first(n);
        for (int i = 0; i < n; ++i) first[i] = i;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(first.begin(), first.end(), g);

        std::vector<std::thread> threads;
        threads.reserve(n);

        auto worker = [&](int m) {
          int base = m * n;
          int first_choice = first[m];
          for (int i = 0; i < n; ++i)
            pl[base + i] = (first_choice + i) % n;
          std::shuffle(pl.begin() + base + 1, pl.begin() + base + n, g);
        };

        for (int m = 0; m < n; ++m)
          threads.emplace_back(worker, m);
        for (auto& th : threads) th.join();
      }

      // ======================= RANDOM =============================
      static void GeneratePrefListsRandom(PreferenceLists& pl, int n, int group_size) {
        Vector<int> ids(n);
        for (int i = 0; i < n; ++i) ids[i] = i;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(ids.begin(), ids.end(), g);

        const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
        const int rows_per_thread = (n + (int)maxThreads - 1) / (int)maxThreads;

        std::vector<std::thread> threads;
        threads.reserve(maxThreads);

        auto worker = [&](int r0, int r1) {
          for (int r = r0; r < r1; ++r) {
            int base = r * n;
            int num_groups = (n + group_size - 1) / group_size;

            for (int gidx = 0; gidx < num_groups; ++gidx) {
              int start = gidx * group_size;
              int end = std::min(start + group_size, n);

              Vector<int> cur_ids(end - start);
              for (int i = start; i < end; ++i)
                cur_ids[i - start] = ids[i];

              std::mt19937 rng((uint32_t)std::random_device{}());
              std::shuffle(cur_ids.begin(), cur_ids.end(), rng);

              for (int i = start; i < end; ++i)
                pl[base + i] = cur_ids[i - start];
            }
          }
        };

        for (unsigned t = 0; t < maxThreads; ++t) {
          int r0 = (int)t * rows_per_thread;
          int r1 = std::min(r0 + rows_per_thread, n);
          if (r0 >= r1) break;
          threads.emplace_back(worker, r0, r1);
        }
        for (auto& th : threads) th.join();
      }

      // ======================= SOLO (MEN) =========================
      static void GeneratePrefListsManSolo(PreferenceLists& pl, int n) {
        auto GenerateRow = [&](int m) {
          int base = m * n;
          int original_m = m;
          if (original_m == n - 1) m = n - 2;

          int w1 = m;
          int w2 = (m - 1 + n - 1) % (n - 1);
          int w_last = n - 1;

          // anchors
          pl[base + 0] = w1;
          pl[base + (n - 2)] = w2;
          pl[base + (n - 1)] = w_last;

          int w = 0, rank = 1;
          while (rank < n - 2) {
            while (w == w1 || w == w2) ++w;
            pl[base + rank] = w;
            ++w;
            ++rank;
          }

          std::random_device rd;
          std::mt19937 g(rd());
          std::shuffle(pl.begin() + base + 1, pl.begin() + base + (n - 2), g);
        };

        const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
        const int rows_per_thread = (n + (int)maxThreads - 1) / (int)maxThreads;

        std::vector<std::thread> threads;
        threads.reserve(maxThreads);

        auto worker = [&](int r0, int r1) {
          for (int r = r0; r < r1; ++r)
            GenerateRow(r);
        };

        for (unsigned t = 0; t < maxThreads; ++t) {
          int r0 = (int)t * rows_per_thread;
          int r1 = std::min(r0 + rows_per_thread, n);
          if (r0 >= r1) break;
          threads.emplace_back(worker, r0, r1);
        }
        for (auto& th : threads) th.join();
      }


      // ======================= SOLO (WOMEN) =======================
      static void GeneratePrefListsWomanSolo(PreferenceLists& pl, int n) {
        auto GenerateRow = [&](int w) {
          int base = w * n;

          if (w < n - 1) {
            int m1 = (w + 1) % (n - 1);
            int m2 = (m1 + n - 1) % n;

            pl[base + 0] = m1;
            pl[base + 1] = m2;

            int m = 0, rank = 2;
            while (rank < n) {
              while (m == m1 || m == m2) ++m;
              pl[base + rank] = m;
              ++m;
              ++rank;
            }

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(pl.begin() + base + 2, pl.begin() + base + n, g);
          }
          else {
            for (int rank = 0; rank < n; ++rank)
              pl[base + rank] = n - 1 - rank;

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(pl.begin() + base, pl.begin() + base + n, g);
          }
        };

        const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
        const int rows_per_thread = (n + (int)maxThreads - 1) / (int)maxThreads;

        std::vector<std::thread> threads;
        threads.reserve(maxThreads);

        auto worker = [&](int r0, int r1) {
          for (int r = r0; r < r1; ++r)
            GenerateRow(r);
        };

        for (unsigned t = 0; t < maxThreads; ++t) {
          int r0 = (int)t * rows_per_thread;
          int r1 = std::min(r0 + rows_per_thread, n);
          if (r0 >= r1) break;
          threads.emplace_back(worker, r0, r1);
        }
        for (auto& th : threads) th.join();
      }

      // ======================= SAVE / LOAD ========================
      static inline String GetWorkloadFilename(WorkloadType type, size_t n, size_t group_size) {
        OutStringStream oss;
        oss << "smp_workload_";
        switch (type) {
        case RANDOM: oss << "random_" << n << "_" << group_size;
          break;
        case SOLO: oss << "solo_" << n;
          break;
        case CONGESTED: oss << "congested_" << n;
          break;
        case PERFECT: oss << "perfect_" << n;
          break;
        }
        oss << ".txt";
        return oss.str();
      }

      static inline void SaveSmpWorkload(const String& filename,
                                         const PreferenceLists& plM,
                                         const PreferenceLists& plW,
                                         int n) {
        const String fullpath = workloadDir + filename;
        OutFStream ofs(fullpath);
        ofs << n << "\n";
        for (int r = 0; r < n; ++r) {
          int base = r * n;
          for (int c = 0; c < n; ++c) ofs << plM[base + c] << " ";
          ofs << "\n";
        }
        for (int r = 0; r < n; ++r) {
          int base = r * n;
          for (int c = 0; c < n; ++c) ofs << plW[base + c] << " ";
          ofs << "\n";
        }
      }

      static inline void LoadSmpWorkload(const String& filename,
                                         PreferenceLists& plM,
                                         PreferenceLists& plW,
                                         int& n_out) {
        const String fullpath = workloadDir + filename;
        InFStream ifs(fullpath);
        if (!ifs.is_open())
          LOG(FATAL) << "Cannot open file: " << fullpath;
        int n;
        ifs >> n;
        n_out = n;
        plM.assign(n * n, 0);
        plW.assign(n * n, 0);
        for (int r = 0; r < n; ++r)
          for (int c = 0; c < n; ++c) ifs >> plM[r * n + c];
        for (int r = 0; r < n; ++r)
          for (int c = 0; c < n; ++c) ifs >> plW[r * n + c];
      }
    };
  };


  // namespace dev {
  //   struct SmpWorkloadDevView;
  //
  //   // Device Resource Manager
  //   struct SmpWorkload {
  //     DArray<int> pref_lists_m;
  //     DArray<int> pref_lists_w;
  //     int n;
  //
  //     explicit SmpWorkload(const ::bsmp::SmpWorkload& smp);
  //
  //     auto DeviceView() -> SmpWorkloadDevView;
  //   };
  //
  //   // Device Resource Handler
  //   struct SmpWorkloadDevView {
  //     DArrayView<int> pref_lists_m;
  //     DArrayView<int> pref_lists_w;
  //     int n;
  //   };
  // }
} // namespace bsmp

// Helper methods to print out a Smp workload
inline std::ostream& operator<<(std::ostream& os, const bsmp::SmpWorkload& smp) {
  std::ostringstream ss;
  int n = smp.n;

  ss << "Preference Lists of Men:\n";
  for (int m = 0; m < n; ++m) {
    // ss << "M" << std::setw(2) << std::setfill('0') << m << ": ";
    ss << "M" << m << ": ";
    int base = m * n;
    for (int w = 0; w < n; ++w) {
      ss << "W" << smp.pref_lists_m[base + w];
      if (w != n - 1) ss << ", ";
    }
    ss << "\n";
  }

  ss << "\nPreference Lists of Women:\n";
  for (int w = 0; w < n; ++w) {
    ss << "W" << w << ": ";
    int base = w * n;
    for (int m = 0; m < n; ++m) {
      ss << "M" << smp.pref_lists_w[base + m];
      if (m != n - 1) ss << ", ";
    }
    ss << "\n";
  }

  os << ss.str();
  return os;
}
