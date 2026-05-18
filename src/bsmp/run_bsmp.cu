#include "bsmp/run_bsmp.h"

#include "bsmp/bsmp_engine.h"

namespace bsmp {
  auto RunBsmp(const SmpWorkload& input) -> Matching {
    BsmpEngine engine(input);
    return engine.Execute();
  }
}
