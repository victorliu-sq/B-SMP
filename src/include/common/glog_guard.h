#ifndef B_SMP_GLOG_GUARD_H
#define B_SMP_GLOG_GUARD_H

#include "glog/logging.h"

namespace bsmp {
  class GlogGuard {
  public:
    explicit GlogGuard(const char* name, const char* log_dir = "tmp/logs")
      : log_dir_(log_dir) {
      FLAGS_log_dir = log_dir;
      google::InitGoogleLogging(name);
      google::InstallFailureSignalHandler();
    }

    ~GlogGuard() {
      google::ShutdownGoogleLogging();
    }

    // Disable copy and move to ensure one active guard per process
    GlogGuard(const GlogGuard&) = delete;
    GlogGuard& operator=(const GlogGuard&) = delete;
    GlogGuard(GlogGuard&&) = delete;
    GlogGuard& operator=(GlogGuard&&) = delete;

    const std::string& LogDir() const noexcept {
      return log_dir_;
    }

  private:
    std::string log_dir_;
  };
}

#endif //B_SMP_GLOG_GUARD_H
