#ifndef B_SMP_BSMP_ENGINE_H
#define B_SMP_BSMP_ENGINE_H
#include "smp_engine_abs.h"
#include "workloads/workloads.h"

namespace bsmp {
  class BsmpEngine : public AbsSmpEngine {
  public:
    auto GetEngineName() const -> String override;

    // Execute() cannot be overriden

    // Constructor: Move an existing smp into current BsmpEngine
    explicit BsmpEngine(SmpWorkload&& smp);

    // Destructor
    ~BsmpEngine();

  private:
    auto IsPerfect() const -> bool override;

    void CoreF() override;

    auto PostF() -> Matching override;

    // --------------------------------------------------------------
    // Define these as public to use device lambda
  public:
    void InitF() override;

    // Pointer To Implementation
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
  };
}

#endif //B_SMP_BSMP_ENGINE_H
