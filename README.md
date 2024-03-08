# PairTorch

## Requirements

```
git clone -b release https://github.com/lammps/lammps.git
cd lammps
cmake -S cmake -B build -DPKG_PLUGIN=yes -DBUILD_SHARED_LIBS=yes
cmake --build build --parallel 8
cd build
make install
```

## Installation

```
git clone <this repo>
cd PairTorch
cmake -S . -B build -DLAMMPS_SOURCE_DIR=<path/to/lammps/src/>
cmake --build build
ctest --test-dir build
```
