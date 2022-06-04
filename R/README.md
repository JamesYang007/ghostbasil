# R GhostBASIL 

This directory contains the `R` wrapper around `ghostbasil`.
__Note: currently, this R package can only build on Linux/MacOS.__

## Dependencies

__R Packages__:
- [devtools](https://www.r-project.org/nosvn/pandoc/devtools.html)

## Developer Installation

The preferred method to install as a developer is to
run the `R` interpreter and the following commands:

```
library(devtools)
load_all(path_to_package)
```

where `path_to_package` is `"."` if in the current directory. 

To install the package locally:

```
library(devtools)
install()
```

## Usage

To use the package after installation, 
simply load the library and call `ghostbasil` 
(see `?ghostbasil` for the full documentation).

Note that if the package was installed using `load_all`,
the library is already loaded.
Otherwise, the package must be loaded by running `library(ghostbasil)`.
