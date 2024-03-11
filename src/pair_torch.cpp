#include "pair_torch.h"

#include "atom.h"
#include "error.h"

#include <cstring>

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
  auto const ntypes = atom->ntypes;

  if (narg != (3 + ntypes)) {
    error->all(FLERR,
               "Wrong number of arguments for `pair_coeff` command, should be `pair_coeff * * "
               "<model>.pt <type1> <type2> ... <typeN>`.");
  }

  if (std::strcmp(arg[0], "*") != 0 || std::strcmp(arg[1], "*") != 0) {
    error->all(
        FLERR,
        "Wrong numeric atom types for `pair_coeff` command, should be `pair_coeff * * ...`.");
  }

  model = torch::jit::load(arg[2], device);
  model.eval();

  type_map.reserve(ntypes);
  for (auto i = 3; i < narg; ++i) {
    type_map.push_back(utils::inumeric(FLERR, arg[i], false, lmp));
  }
}
