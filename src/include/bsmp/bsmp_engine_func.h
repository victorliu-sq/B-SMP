#ifndef B_SMP_BSMP_ENGINE_FUNC_H
#define B_SMP_BSMP_ENGINE_FUNC_H

#ifndef B_SMP_BSMP_ENGINE_H
#include "bsmp/bsmp_engine.h"
#endif

#include <algorithm>
#include <iostream>
#include <set>
#include <thread>

#include "device/launcher.h"

namespace bsmp {
  inline BsmpEngineRep::BsmpEngineRep(const SmpWorkload& smp)
    : smp_(smp),
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

  inline BsmpEngine::BsmpEngine(const SmpWorkload& smp)
    : BsmpEngineRep(smp) {}

  inline auto BsmpEngine::GetEngineName() const -> String {
    return "BambooSmpEngine";
  }

  template <typename F, typename... Arg>
  inline void BsmpEngine::ExecuteNTasklet(size_t n, F f, Arg... arg) {
    LaunchKernelForEach(main_stream_, n, f, arg...);
  }

  inline void BsmpEngine::AsyncD2HPRMatrix() {
    CUDA_CHECK(cudaMemcpyAsync(
      host_prmtx_.GetRawPtr(),
      dev_prmtx_.GetRawPtr(),
      sizeof(PRNode) * static_cast<size_t>(n_) * n_,
      cudaMemcpyDeviceToHost,
      monitor_stream_.cuda_stream()
    ));
    monitor_stream_.Sync();
  }

  inline void BsmpEngine::AsyncD2HPartnerRank() {
    CUDA_CHECK(cudaMemcpyAsync(
      host_partner_rank_.GetRawPtr(),
      dev_partner_rank_.GetRawPtr(),
      sizeof(int) * n_,
      cudaMemcpyDeviceToHost,
      monitor_stream_.cuda_stream()
    ));
    monitor_stream_.Sync();
  }

  inline void BsmpEngine::AsyncD2HNext() {
    CUDA_CHECK(cudaMemcpyAsync(
      host_next_proposed_w_.GetRawPtr(),
      dev_next_proposed_w_.GetRawPtr(),
      sizeof(int) * static_cast<size_t>(n_),
      cudaMemcpyDeviceToHost,
      monitor_stream_.cuda_stream()
    ));
    monitor_stream_.Sync();
  }

  inline auto BsmpEngine::IsPerfect() const -> bool {
    std::set<int> top_choices;

    for (int m = 0; m < n_; ++m) {
      int top_choice = smp_.pref_lists_m[m * n_];
      top_choices.insert(top_choice);
    }

    is_perfect_ = (static_cast<int>(top_choices.size()) == n_);
    return is_perfect_;
  }

  inline void BsmpEngine::InitF() {
    const int n = n_;

    dev_partner_rank_.Fill(n);
    dev_next_proposed_w_.Fill(0);

    std::fill(host_partner_rank_.begin(), host_partner_rank_.end(), n);
    std::fill(host_next_proposed_w_.begin(), host_next_proposed_w_.end(), 0);

    auto pref_list_w_dview = dev_pref_lists_w_.DeviceView();
    auto rank_mtx_w_dview = dev_rank_mtx_w_.DeviceView();

    ExecuteNTasklet(n * n, [=] __device__(size_t tid) mutable {
      int w_idx = tid / n;
      int m_rank = tid % n;

      int m_idx = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, m_rank)];
      rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)] = m_rank;
    });

    auto pref_list_m_dview = dev_pref_lists_m_.DeviceView();
    auto prmtx_dview = dev_prmtx_.DeviceView();

    ExecuteNTasklet(n * n, [=] __device__(size_t tid) mutable {
      int m_idx = tid / n;
      int w_rank = tid % n;

      int w_idx = pref_list_m_dview[IDX_MUL_ADD(m_idx, n, w_rank)];
      int m_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)];

      prmtx_dview[IDX_MUL_ADD(m_idx, n, w_rank)] = {w_idx, m_rank};
    });
  }

  inline void BsmpEngine::DoWorkOnGpu() {
    const int n = n_;
    auto prmtx_dview = dev_prmtx_.DeviceView();
    auto pref_list_w_dview = dev_pref_lists_w_.DeviceView();
    auto next_proposed_w_dview = dev_next_proposed_w_.DeviceView();
    auto partner_rank_dview = dev_partner_rank_.DeviceView();

    ExecuteNTasklet(n_, [=] __device__(size_t tid) mutable {
      int mi = tid;
      int w_rank = 0;
      int w_idx, mi_rank, mj_rank;
      PRNode node;
      bool paired = false;
      while (!paired) {
        node = prmtx_dview[mi * n + w_rank];
        w_idx = node.idx_;
        mi_rank = node.rank_;
        w_rank += 1;

        mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);
        if (mj_rank > mi_rank) {
          next_proposed_w_dview[mi] = w_rank;
          if (mj_rank == n) {
            paired = true;
          }
          else {
            mi = pref_list_w_dview[w_idx * n + mj_rank];
            w_rank = next_proposed_w_dview[mi];
          }
        }
      }
    });
  }

  inline void BsmpEngine::LAProcedure(int m) {
    int w_idx, m_rank, m_idx, w_rank, p_rank;
    m_idx = m;
    w_rank = 0;
    PRNode temp_node{};
    bool is_matched = false;

    while (!is_matched) {
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

        m_idx = smp_.pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
        w_rank = host_next_proposed_w_[m_idx];
      }
      else {
        w_rank++;
      }
    }
  }

  inline void BsmpEngine::MonitorProceture() {
    int it = 0;
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
        break;
      }
      if (unmatched_num_ == 1 && !encountered_1_once) {
        encountered_1_once = true;
        continue;
      }
      if (unmatched_num_ <= 1 && encountered_1_once) {
        break;
      }
      it++;
    }
    while (unmatched_num_ > 1);
  }

  inline void BsmpEngine::DoWorkOnCpu() {
    MonitorProceture();
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

  inline void BsmpEngine::CoreF() {
    std::thread thread_gpu(&BsmpEngine::DoWorkOnGpu, this);
    std::thread thread_cpu(&BsmpEngine::DoWorkOnCpu, this);
    thread_cpu.join();
    thread_gpu.detach();
  }

  inline auto BsmpEngine::PostF() -> Matching {
    const int n = n_;
    Matching match_vec(n);

    if (is_perfect_) {
      for (int m = 0; m < n; ++m) {
        int top_choice = smp_.pref_lists_m[m * n];
        match_vec[m] = top_choice;
      }
    }
    else {
      AsyncD2HPartnerRank();
      for (int w = 0; w < n; ++w) {
        int m_rank = host_partner_rank_[w];
        int m = smp_.pref_lists_w[w * n + m_rank];
        match_vec[m] = w;
      }
    }

    std::cout << "[BsmpEngine::PostF] Final matching computed." << std::endl;
    return match_vec;
  }
}

#endif //B_SMP_BSMP_ENGINE_FUNC_H
