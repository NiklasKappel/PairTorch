#include "lammpsplugin.h"
#include "version.h"

#include "pair_torch.h"

using namespace LAMMPS_NS;

static Pair *pair_torch_creator(LAMMPS *lmp)
{
  return new PairTorch(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;
  lammpsplugin_t plugin;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "torch";
  plugin.info = "PairTorch pair style plugin v1.0";
  plugin.author = "Niklas Kappel (niklas.kappel@kit.edu)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &pair_torch_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
