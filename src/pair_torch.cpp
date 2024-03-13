#include "pair_torch.h"

#include "atom.h"
#include "error.h"
#include "memory.h"
#include "neigh_list.h"

#include <cstring>

#include <torch/script.h>

using namespace LAMMPS_NS;

PairTorch::PairTorch(LAMMPS *lmp) : Pair(lmp) {
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

PairTorch::~PairTorch() {
  if (allocated != 0) {
    memory->destroy(cutsq);
    memory->destroy(setflag);
  }
}

void PairTorch::allocate() {
  allocated = 1;
  auto const np1 = atom->ntypes + 1;
  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(setflag, np1, np1, "pair:setflag");
  for (auto i = 1; i < np1; ++i) {
    for (auto j = i; j < np1; ++j) {
      setflag[i][j] = 0;
    }
  }
}

void PairTorch::compute(int eflag, int vflag) {
  // TODO(niklas): What to do with these?
  ev_init(eflag, vflag);

  // Number of atoms with neighbor lists.
  // This can not be greater than the number of owned atoms.
  // It can be smaller, e.g. if a hybrid pair style is used.
  auto const inum = list->inum;
  // Indices of owned atoms with neighbor lists.
  auto const *const ilist = list->ilist;
  // Neighbor lists.
  auto const *const *const firstneigh = list->firstneigh;
  // Sizes of neighbor lists.
  auto const *const numneigh = list->numneigh;
  // Positions of owned and ghost atoms.
  auto const *const *const x = atom->x;

  int i, j, ii, jj, jnum;                         // NOLINT
  int const *jlist;                               // NOLINT
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq; // NOLINT
  auto const global_cutoff_sq = global_cutoff * global_cutoff;

  auto edge_index_row_1 = std::vector<int>{};
  auto edge_index_row_2 = std::vector<int>{};

  for (ii = 0; ii < inum; ++ii) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    for (jj = 0; jj < jnum; ++jj) {
      j = jlist[jj];
      j &= NEIGHMASK; // NOLINT(hicpp-signed-bitwise): Required by LAMMPS.
      // Now, j is the index of an owned or ghost atom.

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      // The value of rsq is in [0, (global cutoff + skin distance)^2].
      if (rsq < global_cutoff_sq) {
        edge_index_row_1.insert(edge_index_row_1.end(), {i, j});
        edge_index_row_2.insert(edge_index_row_2.end(), {j, i});
      }
    }
  }

  auto const edge_index = torch::tensor(
      {edge_index_row_1, edge_index_row_2},
      torch::dtype(torch::kInt32).device(torch::kCPU));

  // Number of owned and ghost atoms.
  auto const ntotal = atom->nlocal + atom->nghost;
  // Types of owned and ghost atoms.
  auto *const type = atom->type;

  auto const types = torch::zeros(
      {ntotal}, torch::dtype(torch::kInt32).device(torch::kCPU));
  auto const positions = torch::zeros(
      {ntotal, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));

  auto type_accessor = types.accessor<int, 1>();
  auto position_accessor = positions.accessor<float, 2>();

  for (auto k = 0; k < ntotal; ++k) {
    type_accessor[k] = type_map[type[k] - 1];
    position_accessor[k][0] = static_cast<float>(x[k][0]);
    position_accessor[k][1] = static_cast<float>(x[k][1]);
    position_accessor[k][2] = static_cast<float>(x[k][2]);
  }

  // TODO(niklas): Double check if the tensors are correct.
  // Make a test system and print arrays and tensors.

  auto const model_inputs = std::vector<torch::jit::IValue>{
      types.to(device), positions.to(device), edge_index.to(device)};
  auto const model_outputs = model.forward(model_inputs).toTensorVector();

  // TODO(niklas): Move tensors to CPU.
  auto energy_accessor = model_outputs[0].accessor<float, 2>();
  auto forces_accessor = model_outputs[1].accessor<float, 2>();

  // Forces on owned and ghost atoms.
  auto *const *const f = atom->f;

  // TODO(niklas): Test forces.
  // TODO(niklas): Newton flag.
  for (auto k = 0; k < ntotal; ++k) {
    f[k][0] += forces_accessor[k][0];
    f[k][1] += forces_accessor[k][1];
    f[k][2] += forces_accessor[k][2];
  }
}

void PairTorch::settings(int narg, char **arg) {
  if (narg != 1) {
    error->all(
        FLERR, "Wrong number of arguments for `pair_style` command, "
               "should be `pair_style torch "
               "<global_cutoff>`.");
  }

  global_cutoff = utils::numeric(FLERR, arg[0], false, lmp);
}

void PairTorch::coeff(int narg, char **arg) {
  auto const ntypes = atom->ntypes;

  if (narg != (3 + ntypes)) {
    error->all(
        FLERR, "Wrong number of arguments for `pair_coeff` command, "
               "should be `pair_coeff * * "
               "<model>.pt <type1> <type2> ... <typeN>`.");
  }

  if (std::strcmp(arg[0], "*") != 0 || std::strcmp(arg[1], "*") != 0) {
    error->all(
        FLERR, "Wrong numeric atom types for `pair_coeff` command, "
               "should be `pair_coeff * * ...`.");
  }

  model = torch::jit::load(arg[2], device);
  model.eval();

  type_map.reserve(ntypes);
  for (auto i = 3; i < narg; ++i) {
    type_map.push_back(utils::inumeric(FLERR, arg[i], false, lmp));
  }

  if (allocated == 0) {
    allocate();
  }
  for (auto i = 1; i <= ntypes; ++i) {
    for (auto j = i; j <= ntypes; ++j) {
      setflag[i][j] = 1;
    }
  }
}

auto PairTorch::init_one(int i, int j) -> double {
  if (setflag[i][j] == 0) {
    error->all(
        FLERR, "There are LAMMPS atom types with undefined model types.");
  }

  return global_cutoff; // Determines neighbor list cutoff.
}
