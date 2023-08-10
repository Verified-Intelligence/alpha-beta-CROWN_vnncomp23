CPLEX Cuts
===================

A C++ example program to dump cuts from CPLEX solver. There is no existing python interface for this functionality.

In this example I have change CPLEX parameters to tell it to focus on cuts without branching, and without looking for primal solutions. The cut strength is similar to Gurobi and much stronger than scip.

Install CPLEX
-------

It's free for academic use, but the download procedure is complicated. So I uploaded the latest version here:

```
wget http://d.huan-zhang.com/storage/programs/cplex_studio2210.linux_x86_64.bin
chmod +x cplex_studio2210.linux_x86_64.bin
./cplex_studio2210.linux_x86_64.bin
```

Usage
-------

First change the cplex installation path in `Makefile`. Then run `make` to compile.

Then run `./get_cuts`.

Example: `./get_cuts oval21_point5_label1.mps oval_cuts`

The input .mps file is generated from a oval21 benchmark datapoint. It is a standard format of saving a LP/MIP problem in human-readable form and our current Gurobi building code can generate it. See example here on how it is generated: https://github.com/KaidiXu/CROWN-GENERAL/commit/60be115ed3232250fdb95102a882b9feef5efb6e (command included in commit message).

Two output files will be saved: `oval_cuts.indx` and `oval_cuts.cuts`. `oval_cuts.indx` is only updated once, which contains the variable ID to variable name mapping. `oval_cuts.cuts` contains coefficients for all cuts.

You can also generate the corresponding `.mps` file with all cuts added, by uncomment code here: https://github.com/huanzhang12/CPLEX_cuts/blob/739675fa149844972cc348574a6a44ca12a59a94/get_cuts.cpp#L217-L218
It is used for checking the correctness of cuts because it is human-readable, but the generated file is very large and cannot be efficiently parsed in python.


How to interpret results?
-------

Look at the generated `oval_cuts.mps`. All the constraints with letter `r`, `i`, `m`, `q`, `L` etc are cuts.
Coefficients are stored in sparse format. For example, you can search for `m1038` for all the coefficients of the 1038-th cut.

The `oval_cuts.indx` and `oval_cuts.cuts` files contain the same information as the `oval_cuts.mps` (~20MB, 300K lines) but they are much smaller and efficient to handle. See source code for definition of these binary files.

TODOs
-------

The easiest way is to generate the .mps file using our existing gurobi code and
find cuts using this program, see [this example](https://github.com/KaidiXu/CROWN-GENERAL/commit/60be115ed3232250fdb95102a882b9feef5efb6e).
This program should periodically output cuts
which are monitored by our verifier in a separated thread. Then, these cuts are
processed and added during bab.

The mps files may have numerical issues and are very large to read and process.
The next step is to generate the cuts directly in python by adding new python
bindings.

