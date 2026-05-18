#ifndef B_SMP_BSMP_ENGINE_REP_H
#define B_SMP_BSMP_ENGINE_REP_H

#include "device/dev_array.h"
#include "device/stream.h"
#include "workloads/workloads.h"

namespace bsmp {
  struct BsmpEngineRep {
    explicit BsmpEngineRep(const SmpWorkload& smp);

    const SmpWorkload& smp_;
    int n_;

    mutable bool is_perfect_ = false;

    DArray<int> dev_pref_lists_m_;
    DArray<int> dev_pref_lists_w_;
    DArray<int> dev_rank_mtx_w_;
    DArray<PRNode> dev_prmtx_;
    DArray<int> dev_next_proposed_w_;
    DArray<int> dev_partner_rank_;

    HArray<int> host_partner_rank_;
    HArray<int> host_next_proposed_w_;
    HArray<PRNode> host_prmtx_;

    CudaStream main_stream_{};
    CudaStream monitor_stream_{};

    int unmatched_id_ = 0;
    int unmatched_num_ = 0;
  };
}

#endif //B_SMP_BSMP_ENGINE_REP_H
