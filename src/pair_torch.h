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
  void init_style() override;

private:
  void allocate();

  double global_cutoff;
  std::vector<int> type_map;
  torch::Device device;
  torch::jit::Module model;
};

} // namespace LAMMPS_NS

#endif
