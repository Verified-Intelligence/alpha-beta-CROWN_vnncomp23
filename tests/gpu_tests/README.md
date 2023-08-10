## Environment for alpha-beta-crown

Remove old environment if necessary:
```bash
conda deactivate; conda env remove --name alpha-beta-crown
```

Create a new alpha-beta-crown environment:
```bash
conda env create -f complete_verifier/environment.yml
conda activate alpha-beta-crown
```

Get a Gurobi license. For academic users, see [Academic License Registration](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

Download VNNCOMP benchmarks:
```bash
git clone https://github.com/stanleybak/vnncomp2021
git clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks
(cd vnncomp2022_benchmarks && bash setup.sh)
```

Setting the following enrivonment variables may be needed:

```bash
# GCC cannot be too old. If you have a newer version, use that one.
export CC=/opt/rh/devtoolset-11/root/usr/bin/gcc
export CXX=/opt/rh/devtoolset-11/root/usr/bin/g++
export PATH=/opt/rh/devtoolset-11/root/usr/bin/:$PATH

export MKL_THREADING_LAYER=GNU
```

Then, navigate back to this directory (`tests/gpu_tests`).

## Run Testcases

```bash
python test.py -s TESTSET --run
```

* `TESTSET` can be chosen from: `vnncomp`, `vnncomp22`, `beta_crown`.

To run a single benchmark in a test set, use the `-b` option:

```bash
python test.py -b vnncomp/acasxu --run
```

## Check Results

```bash
python test.py -s TESTSET
```

Options:
* `--reference`: Specify a different reference (by default the latest will be taken)
* `--ignore-time`: Ignore comparison on time cost
* `--ignore-decision`: Ignore comparison on branching decisions

## Save a Reference

```bash
python test.py -s TESTSET --install-ref REF_NAME
```

* `REF_NAME` may look like `20211105_41a3`.
