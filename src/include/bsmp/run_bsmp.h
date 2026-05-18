#ifndef B_SMP_RUN_BSMP_H
#define B_SMP_RUN_BSMP_H

#include "common/types.h"
#include "workloads/workloads.h"

namespace bsmp {
  auto RunBsmp(const SmpWorkload& input) -> Matching;
}

#endif //B_SMP_RUN_BSMP_H
