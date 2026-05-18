#ifndef B_SMP_BSMP_ENGINE_H
#define B_SMP_BSMP_ENGINE_H

#include "bsmp/bsmp_engine_rep.h"
#include "smp_engine_abs.h"

namespace bsmp {
  class BsmpEngine : public AbsSmpEngine, private BsmpEngineRep {
  public:
    explicit BsmpEngine(const SmpWorkload& smp);
    ~BsmpEngine() override = default;

    auto GetEngineName() const -> String override;
    void InitF() override;
    void DoWorkOnGpu();

  private:
    auto IsPerfect() const -> bool override;
    void CoreF() override;
    auto PostF() -> Matching override;

    template <typename F, typename... Arg>
    void ExecuteNTasklet(size_t n, F f, Arg... arg);

    void AsyncD2HPRMatrix();
    void AsyncD2HPartnerRank();
    void AsyncD2HNext();
    void DoWorkOnCpu();
    void LAProcedure(int m);
    void MonitorProceture();
  };
}

#include "bsmp/bsmp_engine_func.h"

#endif //B_SMP_BSMP_ENGINE_H
