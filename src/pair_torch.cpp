#include "pair_torch.h"

#include "atom.h"
#include "error.h"

#include <span>
#include <string>

#include <torch/script.h>

using namespace LAMMPS_NS;

PairTorch::PairTorch(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

void PairTorch::compute(int eflag, int vflag) {}

void PairTorch::settings(int narg, char ** /*arg*/)
{
  if (narg > 0) {
    error->all(FLERR,
               "Wrong number of arguments for `pair_style` command, should be `pair_style torch`.");
  }
}

void PairTorch::coeff(int narg, char **arg)
{
  auto const arguments = std::span(arg, narg);

  if (narg != (3 + atom->ntypes)) {
    error->all(FLERR,
               "Wrong number of arguments for `pair_coeff` command, should be `pair_coeff * * "
               "<model>.pt <type1> <type2> ... <typeN>`.");
  }

  if (strcmp(arguments[0], "*") != 0 || strcmp(arguments[1], "*") != 0) {
    error->all(
        FLERR,
        "Wrong numeric atom types for `pair_coeff` command, should be `pair_coeff * * ...`.");
  }

  model = torch::jit::load(arguments[2], device);
  model.eval();

  auto const types = arguments.subspan(3);
  type_map.reserve(types.size());
  for (auto *type : types) { type_map.push_back(std::stoi(type)); }
}
