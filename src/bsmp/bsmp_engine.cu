#include "bsmp/bsmp_engine.h"

#include <set>

#include "device/stream.h"
#include "device/launcher.h"

namespace bsmp {
  struct BsmpEngine::Impl {
    SmpWorkload smp_;
    int n_;

    // Perfect Flag
    bool is_perfect_ = false;

    // GPU (Device) data
    DArray<int> dev_pref_lists_m_;
    DArray<int> dev_pref_lists_w_;
    DArray<int> dev_rank_mtx_w_;
    DArray<PRNode> dev_prmtx_;
    DArray<int> dev_next_proposed_w_;
    DArray<int> dev_partner_rank_;

    // Host (CPU) mirrors
    HArray<int> host_partner_rank_;
    HArray<int> host_next_proposed_w_;
    HArray<PRNode> host_prmtx_;

    // cuda streams
    CudaStream main_stream_{}; // For GPU Kernel launching
    CudaStream monitor_stream_{}; // for async CPU monitoring

    // Hybrid monitoring
    int unmatched_id_ = 0;
    int unmatched_num_ = 0;

    explicit Impl(SmpWorkload&& smp)
      : smp_(std::move(smp)),
        n_(smp_.n),
        dev_pref_lists_m_(n_ * n_),
        dev_pref_lists_w_(n_ * n_),
        dev_rank_mtx_w_(n_ * n_),
        dev_prmtx_(n_ * n_),
        dev_next_proposed_w_(n_),
        dev_partner_rank_(n_),
        host_partner_rank_(n_),
        host_next_proposed_w_(n_),
        host_prmtx_(n_ * n_) {
      dev_pref_lists_m_.SetToDevice(smp_.pref_lists_m);
      dev_pref_lists_w_.SetToDevice(smp_.pref_lists_w);
    }

    template <typename F, typename... Arg>
    void ExecuteNTasklet(size_t n, F f, Arg... arg) {
      LaunchKernelForEach(main_stream_, n, f, arg...);
    }

    void AsyncD2HPRMatrix() {
      CUDA_CHECK(cudaMemcpyAsync(
        host_prmtx_.GetRawPtr(),
        dev_prmtx_.GetRawPtr(),
        sizeof(PRNode) * static_cast<size_t>(n_) * n_,
        cudaMemcpyDeviceToHost,
        monitor_stream_.cuda_stream()
      ));
      monitor_stream_.Sync();
    }

    void AsyncD2HPartnerRank() {
      CUDA_CHECK(cudaMemcpyAsync(
        host_partner_rank_.GetRawPtr(),
        dev_partner_rank_.GetRawPtr(),
        sizeof(int) * n_,
        cudaMemcpyDeviceToHost,
        monitor_stream_.cuda_stream()
      ));
      monitor_stream_.Sync();
    }

    void AsyncD2HNext() {
      CUDA_CHECK(cudaMemcpyAsync(
        host_next_proposed_w_.GetRawPtr(),
        dev_next_proposed_w_.GetRawPtr(),
        sizeof(int) * static_cast<size_t>(n_),
        cudaMemcpyDeviceToHost,
        monitor_stream_.cuda_stream()
      ));
      monitor_stream_.Sync();
    }

    void DoWorkOnGpu() {
      // Device views for all necessary arrays
      const int n = n_;
      auto prmtx_dview = this->dev_prmtx_.DeviceView();
      auto pref_list_w_dview = this->dev_pref_lists_w_.DeviceView();
      auto next_proposed_w_dview = this->dev_next_proposed_w_.DeviceView();
      auto partner_rank_dview = this->dev_partner_rank_.DeviceView();

      // Launch one thread per man
      this->ExecuteNTasklet(n_, [=] __device__(size_t tid) mutable {
        int mi = tid; // man index
        int w_rank = 0; // current woman rank index
        int w_idx, mi_rank, mj_rank;
        PRNode node;
        bool paired = false;
        while (!paired) {
          // Get the PRNode corresponding to (mi, w_rank)
          // if (mi * n + w_rank < n * n) {
          //   printf("Thread %zu: mi=%d, w_rank=%d\n", tid, mi, w_rank);
          // }
          node = prmtx_dview[mi * n + w_rank];
          w_idx = node.idx_;
          mi_rank = node.rank_;
          w_rank += 1;
          // Attempt to atomically update partner_rank for this woman
          mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);

          // If this man was accepted (his rank is lower = better)
          if (mj_rank > mi_rank) {
            // Record that mi has proposed up to this woman
            next_proposed_w_dview[mi] = w_rank;

            // If woman was free, pairing succeeds
            if (mj_rank == n) {
              paired = true;
            }
            else {
              // Otherwise, her old partner becomes free again (man index lookup)
              mi = pref_list_w_dview[w_idx * n + mj_rank];
              // Resume from his last proposed woman
              w_rank = next_proposed_w_dview[mi];
            }
          }
        }
      });
    }

    void DoWorkOnCpu() {
      MonitorProceture(); // exit when unmatched_num is either 0 or 1.
      if (unmatched_num_ == 1) {
        AsyncD2HPRMatrix();
        AsyncD2HNext();

        LOG(INFO) << "CPU starts LAProcedure.";
        LAProcedure(unmatched_id_);
        LOG(INFO) << "CheckKernel (CPU) won the contention.";
      }
      else {
        LOG(INFO) << "CheckKernel (GPU) won the contention.";
      }
    }

    void LAProcedure(int m) {
      int w_idx, m_rank, m_idx, w_rank, p_rank;
      m_idx = m;
      w_rank = 0;
      PRNode temp_node{};
      bool is_matched = false;
      int iterations = 0;

      while (!is_matched) {
        iterations += 1;
        // temp_node = host_prnodes_m_view_[m_idx * n_ + w_rank];
        // temp_node = host_prmtx_[IDX_MUL_ADD(m_idx, n_, w_rank)];
        // temp_node = host_prmtx_[IDX_MUL_ADD(m_idx, n_, w_rank)];
        temp_node = host_prmtx_[IDX_MUL_ADD(m_idx, n_, w_rank)];

        w_idx = temp_node.idx_;
        m_rank = temp_node.rank_;
        p_rank = host_partner_rank_[w_idx];
        if (p_rank == n_) {
          host_next_proposed_w_[m_idx] = w_rank;
          host_partner_rank_[w_idx] = m_rank;
          is_matched = true;
        }
        else if (p_rank > m_rank) {
          host_next_proposed_w_[m_idx] = w_rank;
          host_partner_rank_[w_idx] = m_rank;

          // m_idx = pref_lists_w_view_[w_idx * n_ + p_rank];
          m_idx = smp_.pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
          w_rank = host_next_proposed_w_[m_idx];
        }
        else {
          w_rank++;
        }
      }
    }

    void MonitorProceture() {
      int it = 0;
      // const int total = n_ * (n_ - 1) / 2;
      const size_t total = SIZE_MUL(n_, (n_ - 1)) / 2;
      bool encountered_1_once = false;

      do {
        SLEEP_MILLISECONDS(10);
        AsyncD2HPartnerRank();

        unmatched_id_ = total;
        unmatched_num_ = 0;
        for (int w = 0; w < n_; w++) {
          if (host_partner_rank_[w] == n_) {
            unmatched_num_++;
          }
          else {
            int m_rank = host_partner_rank_[w];
            unmatched_id_ -= smp_.pref_lists_w[IDX_MUL_ADD(w, n_, m_rank)];
          }
        }

        if (unmatched_num_ == 0) {
          break; // Exit immediately if 0 unmatched
        }
        if (unmatched_num_ == 1 && !encountered_1_once) {
          encountered_1_once = true;
          continue; // Go into second observation loop
        }
        if (unmatched_num_ <= 1 && encountered_1_once) {
          break; // Second observation loop: exit no matter if 0 or 1
        }
        it++;
      }
      while (unmatched_num_ > 1);
    }
  };


  BsmpEngine::BsmpEngine(SmpWorkload&& smp)
    : pimpl_(std::make_unique<Impl>(std::move(smp))) {}

  BsmpEngine::~BsmpEngine() = default;


  auto BsmpEngine::GetEngineName() const -> String {
    return "BambooSmpEngine";
  }

  // Private Member functions
  auto BsmpEngine::IsPerfect() const -> bool {
    auto& self = *pimpl_;

    std::set<int> top_choices;

    for (int m = 0; m < self.n_; ++m) {
      int top_choice = self.smp_.pref_lists_m[m * self.n_]; // man's first preferred woman
      top_choices.insert(top_choice);
    }

    // Store the result into Impl for reuse
    self.is_perfect_ = (static_cast<int>(top_choices.size()) == self.n_);

    return self.is_perfect_;
  }

  void BsmpEngine::InitF() {
    auto& self = *pimpl_;

    const int n = self.n_;

    // ==============================================================
    // 1. Initialize metadata
    // ==============================================================
    self.dev_partner_rank_.Fill(n); // Each woman free (rank n means unassigned)
    self.dev_next_proposed_w_.Fill(0); // Each man starts with w_rank = 0

    // Initialize host metadata too
    std::fill(self.host_partner_rank_.begin(), self.host_partner_rank_.end(), n);
    std::fill(self.host_next_proposed_w_.begin(), self.host_next_proposed_w_.end(), 0);

    // ==============================================================
    // 2. Initialize Rank Matrix on device
    // ==============================================================
    auto pref_list_w_dview = self.dev_pref_lists_w_.DeviceView();
    auto rank_mtx_w_dview = self.dev_rank_mtx_w_.DeviceView();

    self.ExecuteNTasklet(n * n, [=] __device__(size_t tid) mutable {
      int w_idx = tid / n;
      int m_rank = tid % n;

      int m_idx = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, m_rank)];
      rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)] = m_rank;
    });

    // ==============================================================
    // 3. Initialize PRMatrix on device
    // ==============================================================
    auto pref_list_m_dview = self.dev_pref_lists_m_.DeviceView();
    auto prmtx_dview = self.dev_prmtx_.DeviceView();

    self.ExecuteNTasklet(n * n, [=] __device__(size_t tid) mutable {
      int m_idx = tid / n;
      int w_rank = tid % n;

      int w_idx = pref_list_m_dview[IDX_MUL_ADD(m_idx, n, w_rank)];
      int m_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)];

      prmtx_dview[IDX_MUL_ADD(m_idx, n, w_rank)] = {w_idx, m_rank};
    });
  }

  void BsmpEngine::CoreF() {
    std::thread thread_gpu(&Impl::DoWorkOnGpu, pimpl_.get());
    std::thread thread_cpu(&Impl::DoWorkOnCpu, pimpl_.get());
    thread_cpu.join();
    thread_gpu.detach();

    // thread_gpu.join();
  }

  auto BsmpEngine::PostF() -> Matching {
    auto& self = *pimpl_;
    const int n = self.n_;
    Matching match_vec(n);

    // Use cached flag instead of recomputing
    if (self.is_perfect_) {
      for (int m = 0; m < n; ++m) {
        int top_choice = self.smp_.pref_lists_m[m * n];
        match_vec[m] = top_choice;
      }
    }
    else {
      self.AsyncD2HPartnerRank();
      for (int w = 0; w < n; ++w) {
        int m_rank = self.host_partner_rank_[w];
        int m = self.smp_.pref_lists_w[w * n + m_rank];
        match_vec[m] = w;
      }
    }

    std::cout << "[BsmpEngine::PostF] Final matching computed." << std::endl;
    return match_vec;
  }
}
