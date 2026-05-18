#include "flag.h"

#include "bsmp/run_bsmp.h"
#include "common/glog_guard.h"

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  bsmp::GlogGuard glog_guard("bsmp_exe", "tmp/logs"); // RAII wrapper for GLog


  bsmp::SmpWorkload wkld(FLAGS_workload_size, FLAGS_workload_type, false);
  bsmp::Matching matching = bsmp::RunBsmp(wkld);

  std::cout << "Final Matching Results:\n";
  for (size_t m = 0; m < matching.size(); ++m) {
    std::cout << "  Man M" << m << " <-> Woman W" << matching[m] << "\n";
  }
  std::_Exit(0);
}
