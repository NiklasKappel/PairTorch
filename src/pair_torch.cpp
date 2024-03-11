#include "pair_torch.h"

#include "atom.h"
#include "error.h"
#include "neigh_list.h"

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

// TODO(niklas):
// The `positions` and `types` tensors contains positions and types of all local atoms.
// The `edge_index` tensor contains local indices of atoms described by the ML pair style only.
// Passing these tensors into the model means the model must be invariant under the addition of unconnected atoms.
// For ML/MM, it also means we crunch much larger tensors than necessary.
// Alternatively, one could translate back and forth between local and ML indices and pass smaller `positions` and `types` tensors.

void PairTorch::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Types of all owned and ghost atoms in the local subdomain.
  auto const *const type = atom->type;

  // Positions of all owned and ghost atoms in the local subdomain.
  auto const *const *const x = atom->x;

  // Number of atoms for which neighbor lists have been created.
  // This can not be greater than the number of owned atoms.
  // It can be smaller, e.g. if a hybrid pair style is used.
  auto const inum = list->inum;

  // Local indices of all atoms for which neighbor lists have been created.
  auto const *const ilist = list->ilist;

  // Neighbor lists.
  auto const *const *const firstneigh = list->firstneigh;

  // Numbers of entries of all neighbor lists.
  auto const *const numneigh = list->numneigh;

  // Cached loop variables.
  auto const *jlist = firstneigh[0];
  auto jnum = numneigh[0];

  for (int ii = 0, i = 0; ii < inum; ++ii) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (int jj = 0, j = 0; jj < jnum; ++jj) {
      j = jlist[jj];
      j &= NEIGHMASK;    // NOLINT(hicpp-signed-bitwise): Required by LAMMPS.
      // j is now the local index of a neighbor of the atom with local index i.
      // Build the `edge_index` tensor.
    }
  }
}

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
