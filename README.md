# GhostBASIL

This is a repository for experimental code using GhostKnockoff and BASIL framework
for fast and efficient LASSO fitting on high-dimensional data.
Specifically, this library is created with the intention of applying it on GWAS summary statistics.

## Dependencies

- [Bazel](https://docs.bazel.build/versions/main/install.html)

## Developer Installation

Run the following command to generate machine-specific `.bazelrc`:
```
./generate_bazelrc
```

Then, install the developement dependencies:
```
mamba update -y conda
mamba env create
conda activate ghostbasil
```

Our tools require `omp` to be installed on the system.
On Mac, it must be installed with Homebrew as our build system assumes:
```
brew install libomp
```

See the following for installations of each of the sub-packages:
- [R README](R/README.md)
- [C++ README](ghostbasil/README.md)

## References

- [Summary statistics knockoff inference empowers identification of putative causal variants in genome-wide association studies](https://www.biorxiv.org/content/10.1101/2021.12.06.471440v1)
- [A fast and scalable framework for large-scale and ultrahigh-dimensional sparse regression with application to the UK Biobank](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009141)
