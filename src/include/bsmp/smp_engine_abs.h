#ifndef SMP_ENGINE_ABSTRACT_H
#define SMP_ENGINE_ABSTRACT_H
#include "common/stopwatch.h"
#include "common/types.h"
#include "glog/logging.h"

namespace bsmp {
  class AbsSmpEngine {
  public:
    virtual ~AbsSmpEngine() = default;

    virtual auto Execute() -> Matching {
      sw_.Start();
      bool is_perfect = IsPerfect();
      sw_.Stop();
      precheck_time_ = sw_.GetEclapsedMs();

      if (!is_perfect) {
        // Init Phase
        sw_.Start();
        InitF(); // Do the computation
        sw_.Stop();
        init_time_ = sw_.GetEclapsedMs();

        // Core Computation Phase
        sw_.Start();
        CoreF(); // Do the computation
        sw_.Stop();
        core_time_ = sw_.GetEclapsedMs();
      }
      else {
        std::cout << "Perfect matching identified; skipping execution steps." << std::endl;
      }

      // Post-Process
      return PostF();
    }

    virtual void PrintProfilingInfo() {
      std::cout << GetEngineName() << " Profiling Info:" << std::endl;
      std::cout << GetEngineName() << " Precheck Time: " << precheck_time_ << " ms" << std::endl;
      std::cout << GetEngineName() << " Init Time: " << init_time_ << " ms" << std::endl;
      std::cout << GetEngineName() << " Core Time: " << core_time_ << " ms" << std::endl;
      std::cout << GetEngineName() << " Total Time: " << precheck_time_ + init_time_ +
        core_time_
        << " ms" << std::endl;
      std::cout << "==================================================================" << std::endl;

      LOG(INFO) << GetEngineName() << " Profiling Info:";
      LOG(INFO) << GetEngineName() << " Precheck Time: " << precheck_time_ << " ms";
      LOG(INFO) << GetEngineName() << " Init Time: " << init_time_ << " ms";
      LOG(INFO) << GetEngineName() << " Core Time: " << core_time_ << " ms";
      LOG(INFO) << GetEngineName() << " Total Time: " << precheck_time_ + init_time_ +
        core_time_
        << " ms";
      LOG(INFO) << "==================================================================";
    }

  protected:
    virtual auto IsPerfect() const -> bool {
      return false;
    }

    virtual void InitF() = 0;

    virtual void CoreF() = 0;

    virtual auto PostF() -> Matching = 0;

    virtual auto GetEngineName() const -> String = 0;

    ms_t precheck_time_{0};
    ms_t init_time_{0};
    ms_t core_time_{0};

  private:
    StopWatch sw_{false};
  };
}


#endif //SMP_ENGINE_ABS_H
