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
