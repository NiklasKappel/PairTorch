#include "pair_torch.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"

#include <cstring>

#include <torch/script.h>

using namespace LAMMPS_NS;

PairTorch::PairTorch(LAMMPS *lmp)
    : Pair{lmp}, global_cutoff{0.0}, device{torch::kCPU} {
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }
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
  ev_init(eflag, vflag);

  // ==============================
  // Read LAMMPS data.
  // ==============================

  // Number of owned atoms.
  auto const nlocal = atom->nlocal;
  // Number of owned and ghost atoms.
  auto const ntotal = nlocal + atom->nghost;
  // Number of atoms with neighbor lists.
  // This can not be greater than the number of owned atoms.
  // It can be smaller, e.g. if a hybrid pair style is used.
  auto const inum = list->inum;
  // Indices of atoms with neighbor lists.
  // These are all owned atoms.
  auto const *const ilist = list->ilist;
  // Neighbor lists.
  auto const *const *const firstneigh = list->firstneigh;
  // Sizes of neighbor lists.
  auto const *const numneigh = list->numneigh;
  // Types of owned and ghost atoms.
  auto *const type = atom->type;
  // Positions of owned and ghost atoms.
  auto const *const *const x = atom->x;
  // Forces on owned and ghost atoms.
  auto *const *const f = atom->f;

  // ==============================
  // Process neighbor lists.
  // ==============================

  int i, j, ii, jj, jnum;                         // NOLINT
  int const *jlist;                               // NOLINT
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq; // NOLINT
  auto const global_cutoff_sq = global_cutoff * global_cutoff;

  auto edge_indices = std::vector<int>{};

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
        edge_indices.insert(edge_indices.end(), {i, j, j, i});
      }
    }
  }

  // ==============================
  // Build input tensors.
  // ==============================

  auto const edge_index =
      torch::from_blob(
          edge_indices.data(), {static_cast<long>(edge_indices.size() / 2), 2},
          torch::dtype(torch::kInt32).device(torch::kCPU))
          .transpose_(0, 1);

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

  // ==============================
  // Run model.
  // ==============================

  auto const model_inputs = std::vector<torch::jit::IValue>{
      types.to(device), positions.to(device), edge_index.to(device)};
  auto const model_outputs = model.forward(model_inputs).toTuple()->elements();

  auto const forces = torch::Tensor{model_outputs[1].toTensor()}.to(
      torch::kCPU);
  auto const forces_accessor = forces.accessor<float, 2>();

  // ==============================
  // Write LAMMPS data.
  // ==============================

  auto const k_max = force->newton_pair != 0 ? ntotal : nlocal;

  for (auto k = 0; k < k_max; ++k) {
    f[k][0] += forces_accessor[k][0];
    f[k][1] += forces_accessor[k][1];
    f[k][2] += forces_accessor[k][2];
  }

  if (vflag_fdotr != 0) {
    virial_fdotr_compute();
  }
}

void PairTorch::settings(int narg, char **arg) {
  if (narg != 1) {
    error->all(
        FLERR,
        "Wrong number of arguments for `pair_style` command, "
        "should be `pair_style torch "
        "<global_cutoff>`.");
  }

  global_cutoff = utils::numeric(FLERR, arg[0], false, lmp);
}

void PairTorch::coeff(int narg, char **arg) {
  auto const ntypes = atom->ntypes;

  if (narg != (3 + ntypes)) {
    error->all(
        FLERR,
        "Wrong number of arguments for `pair_coeff` command, "
        "should be `pair_coeff * * "
        "<model>.pt <type1> <type2> ... <typeN>`.");
  }

  if (std::strcmp(arg[0], "*") != 0 || std::strcmp(arg[1], "*") != 0) {
    error->all(
        FLERR,
        "Wrong numeric atom types for `pair_coeff` command, "
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
