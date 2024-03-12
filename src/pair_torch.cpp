#include "pair_torch.h"

#include "atom.h"
#include "error.h"
#include "memory.h"
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

PairTorch::~PairTorch()
{
  if (allocated != 0) {
    memory->destroy(cutsq);
    memory->destroy(setflag);
  }
}

// The boilerplate associated with `cutsq` and `setflag` can not be avoided.
// See https://matsci.org/t/what-cutoff-does-the-neighbor-list-use-if-init-one-is-not-defined/54098.
void PairTorch::allocate()
{
  allocated = 1;
  auto const np1 = atom->ntypes + 1;
  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(setflag, np1, np1, "pair:setflag");
  for (auto i = 1; i < np1; ++i) {
    for (auto j = i; j < np1; ++j) { setflag[i][j] = 0; }
  }
}

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
  int i, j, ii, jj, jnum;                            // NOLINT
  int const *jlist;                                  // NOLINT
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;    // NOLINT
  auto const global_cutoff_sq = global_cutoff * global_cutoff;

  for (ii = 0; ii < inum; ++ii) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    for (jj = 0; jj < jnum; ++jj) {
      j = jlist[jj];
      j &= NEIGHMASK;    // NOLINT(hicpp-signed-bitwise): Required by LAMMPS.
      // j is now the local index of a neighbor of the atom with local index i.

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < global_cutoff_sq) {}
    }
  }
}

void PairTorch::settings(int narg, char **arg)
{
  if (narg != 1) {
    error->all(FLERR,
               "Wrong number of arguments for `pair_style` command, should be `pair_style torch "
               "<global_cutoff>`.");
  }

  global_cutoff = utils::numeric(FLERR, arg[0], false, lmp);
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

  if (allocated == 0) { allocate(); }
  for (auto i = 1; i <= ntypes; ++i) {
    for (auto j = i; j <= ntypes; ++j) { setflag[i][j] = 1; }
  }
}

auto PairTorch::init_one(int i, int j) -> double
{
  if (setflag[i][j] == 0) {
    error->all(FLERR, "There are LAMMPS atom types with undefined model types.");
  }

  return global_cutoff;    // Determines neighbor list cutoff.
}
