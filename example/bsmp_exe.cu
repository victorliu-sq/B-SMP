#include "bsmp/run_bsmp.h"
#include "common/glog_guard.h"
#include "workloads/workloads.h"

int main(int argc, char** argv) {
  // RAII wrapper for GLog
  bsmp::GlogGuard glog_guard("bsmp_exe", "tmp/logs");

  // 1. Initialize SMP Workload Size
  const int n = std::stoi(argv[1]);

  // 2. Initialize SMP Workload Type
  std::string type_str = argv[2];
  bsmp::WorkloadType workload_type;
  if (type_str == "RANDOM") {
    workload_type = bsmp::RANDOM;
  }
  else if (type_str == "CONGESTED") {
    workload_type = bsmp::CONGESTED;
  }
  else if (type_str == "SOLO") {
    workload_type = bsmp::SOLO;
  }
  else if (type_str == "PERFECT") {
    workload_type = bsmp::PERFECT;
  }
  else {
    std::cerr << "Error: invalid workload type: " << type_str << "\n";
    return EXIT_FAILURE;
  }

  // 3. Initialize SMP workload
  bsmp::SmpWorkload wkld(n, workload_type, false);
  // 4. Run BsmpEngine on this SMP workload
  bsmp::Matching matching = bsmp::RunBsmp(wkld);

  bsmp::operator<<(std::cout, matching);
  std::_Exit(0);
}
