#ifndef PAIR_TORCH_H
#define PAIR_TORCH_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTorch : public Pair {
 public:
  explicit PairTorch(class LAMMPS *);
  ~PairTorch() override = default;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
};

}    // namespace LAMMPS_NS

#endif
