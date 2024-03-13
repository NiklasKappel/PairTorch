#ifndef PAIR_TORCH_H
#define PAIR_TORCH_H

#include "pair.h"

#include <vector>

#include <torch/torch.h>

namespace LAMMPS_NS {

class PairTorch : public Pair {
public:
  explicit PairTorch(class LAMMPS *);
  ~PairTorch() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  auto init_one(int, int) -> double override;

private:
  void allocate();

  double global_cutoff = 0.0;
  std::vector<int> type_map;
  torch::Device device = torch::kCPU;
  torch::jit::Module model;
};

} // namespace LAMMPS_NS

#endif
