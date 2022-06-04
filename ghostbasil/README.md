# GhostBASIL C++ Core

This directory contains the core C++ code for `GhostBASIL`.
`ghostbasil` is a C++ template header-only library.
The `include/` folder contains the template code and `test/` folder
contains the unittests.

## Build

__To build the library__: there is nothing to do!

__To build tests__:
```
bazel test //ghostbasil/test/... -c dbg
```

__To build benchmarks__:
```
bazel run //ghostbasil/benchmark:name_of_benchmark -c opt
```
where `name_of_benchmark.cpp` in `benchmark/` directory.
