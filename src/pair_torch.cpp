#include "pair_torch.h"

using namespace LAMMPS_NS;

PairTorch::PairTorch(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

void PairTorch::compute(int eflag, int vflag) {}

void PairTorch::settings(int narg, char **arg) {}

void PairTorch::coeff(int narg, char **arg) {}
