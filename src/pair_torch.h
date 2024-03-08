#ifndef PAIR_TORCH_H
#define PAIR_TORCH_H

#include "pair.h"

#include <vector>

#include <torch/torch.h>

namespace LAMMPS_NS {

class PairTorch : public Pair {
 public:
  explicit PairTorch(class LAMMPS *);
  ~PairTorch() override = default;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;

 private:
  torch::Device device{torch::kCPU};
  torch::jit::Module model;
  std::vector<int> type_map;
};

}    // namespace LAMMPS_NS

#endif
