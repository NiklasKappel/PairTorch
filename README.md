# PairTorch

PairTorch is a pair style plugin for [LAMMPS](https://www.lammps.org) that lets you use PyTorch machine learning force fields that follow a few basic [rules](#prepare-your-model).

## Requirements

PairTorch assumes that you build and install LAMMPS from source. Your LAMMPS installation must be able to load [plugins](https://docs.lammps.org/plugin.html). This means that you need to build LAMMPS with the `PKG_PLUGIN` and `BUILD_SHARED_LIBS` options enabled. For example, run the following steps:

```
git clone -b release https://github.com/lammps/lammps.git
cd lammps
cmake -S cmake -B build -DPKG_PLUGIN=yes -DBUILD_SHARED_LIBS=yes
cmake --build build --parallel 8
cd build
make install
```

> [!IMPORTANT]
> The installation instructions for LAMMPS above and PairTorch below install shared libraries (including LibTorch) to a user directory, e.g. `~/.local/lib64/` on UNIX. Make sure this directory is on your runtime search path, e.g. by adding `export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:~/.local/lib64/"` to your `~/.bashrc`.

To run models on GPUs, CUDA 11.8 or newer must be installed on your system.

## Installation

Run the following steps, replacing `<path/to/lammps/src/>` with the path to the `src/` directory in your local copy of the LAMMPS repository:

```
git clone <this repo>
cd PairTorch
cmake -S . -B build -DLAMMPS_SOURCE_DIR=<path/to/lammps/src/>
cmake --build build --parallel 8
cd build
make install
```

## Prepare your model

Your model is expected to have a `forward` method with a signature that follows [PyG](https://pytorch-geometric.readthedocs.io) standards:

```python
class MyModel(torch.nn.Module):
    ...

    def forward(self, types, positions, edge_index, **kwargs):
        """Compute the energy and forces of the system.

        Parameters
        ----------
        types : torch.Tensor
            Atom types with shape [num_atoms]
            and data type torch.int32.
        positions : torch.Tensor
            Atom positions with shape [num_atoms, num_dimensions]
            and data type torch.float32.
        edge_index: torch.Tensor
            Interaction graph connectivity with shape [2, num_edges]
            and data type torch.long.
        kwargs: Any
            Keyword arguments. They must be optional
            and PairTorch will not pass them to your model.

        Returns
        -------
        energy : torch.Tensor
            Total energy of the system with shape [1, 1]
            and data type torch.float32.
        forces : torch.Tensor
            Forces on each atom with shape [num_atoms, num_dimensions]
            and data type torch.float32.
        """
        ...
        return energy, forces
```

To load your model in LAMMPS, you need to compile it with the [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) compiler and save it to a file, for example:

```python
if __name__ == "__main__":
    model = MyModel()

    # Train the model.
    ...

    # Compile and save the model.
    compiled_model = torch.jit.script(model)
    compiled_model.save("my_model.pt")
```

Make sure that your model is trained to output energies and forces in the same units that your LAMMPS input file [specifies](https://docs.lammps.org/units.html).

## Use your model in LAMMPS

Load your model in a LAMMPS input file with the following commands, replacing the placeholders. `<global_cutoff>` is the cutoff distance used to build the `edge_index` tensor from the LAMMPS neighbor list. `<path/to/model.pt>` is the path to your saved model. `<typeX>` is the integer atom type that your model uses and that corresponds to the LAMMPS atom type `X`.

```
plugin load PairTorch.so
pair_style torch <global_cutoff>
pair_coeff * * <path/to/model.pt> <type1> <type2> ... <typeN>
```

## References

PairTorch was inspired by the following model-specific LAMMPS interfaces:

- [pair_allegro](https://github.com/mir-group/pair_allegro)
- [pair_nequip](https://github.com/mir-group/pair_nequip)
- [pair_schnetpack](https://github.com/atomistic-machine-learning/schnetpack/tree/master/interfaces/lammps)
