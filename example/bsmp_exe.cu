#include "bsmp/bsmp_engine.h"
#include "common/glog_guard.h"
#include "workloads/workloads.h"

using namespace bsmp;

int main(int argc, char** argv) {
  // RAII wrapper for GLog
  GlogGuard glog_guard("bsmp_exe", "tmp/logs");

  // 1. Initialize SMP Workload Size
  const int n = std::stoi(argv[1]);

  // 2. Initialize SMP Workload Type
  std::string type_str = argv[2];
  WorkloadType workload_type;
  if (type_str == "RANDOM") {
    workload_type = RANDOM;
  }
  else if (type_str == "CONGESTED") {
    workload_type = CONGESTED;
  }
  else if (type_str == "SOLO") {
    workload_type = SOLO;
  }
  else if (type_str == "PERFECT") {
    workload_type = PERFECT;
  }
  else {
    std::cerr << "Error: invalid workload type: " << type_str << "\n";
    return EXIT_FAILURE;
  }

  // 3. Initialize SMP workload
  SmpWorkload wkld(n, workload_type, false);
  // std::cout << "Testing SOLO workload:\n" << wkld << std::endl;

  // 4. Run BsmpEngine on this SMP workload
  BsmpEngine engine(std::move(wkld));
  Matching matching = engine.Execute();

  std::cout << matching;
  std::_Exit(0);
}
