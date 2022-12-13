# Benchmarks

This benchmark system is similar to `testthat`, 
but uses a Python script as a driver to run the benchmarks.
Simply run
```python
python bench.py <provide-some-arguments>
```
It is recommended to run with Python 3.

The full set of arguments are
```bash
> python bench.py -h

usage: bench.py [-h] [-d [bench_dir]] -r path_to_repo -b baseline_branch -n new_version_branch [-c] [-i include-subset [include-subset ...]]
                [-e exclude-subset [exclude-subset ...]] [--analyze] [--analyze-only]

Benchmark driver

optional arguments:
  -h, --help            show this help message and exit
  -d [bench_dir]        Path to directory containing subdirectory "R/" that contains R benchmark scripts (default: current working directory).
  -r path_to_repo       Path or URL to repository to benchmark.
  -b baseline_branch    Branch or tag name for baseline.
  -n new_version_branch
                        Branch or tag name for new version to benchmark against baseline.
  -c                    If specified, cleans any files that the script produced from previous runs. This is necessary if the repository path/url changed
                        from the previous run.
  -i include-subset [include-subset ...]
                        A list of the subset of benchmarks to run. At least one name must be provided if the flag is passed. The names can either be the
                        full name (e.g. bench_glmnet) or the keyword in the name (e.g. glmnet).
  -e exclude-subset [exclude-subset ...]
                        A list of the subset of benchmarks to exclude from running. At least one name must be provided if the flag is passed. The names can
                        either be the full name (e.g. bench_glmnet) or the keyword in the name (e.g. glmnet).
  --analyze             Analyze also after running the benchmarks. The folder analyze/ containing analyze scripts must exist.
  --analyze-only        Analyze only and do not run the benchmarks. The folder analyze/ containing analyze scripts must exist. If --analyze is also passed,
                        it is ignored, i.e. benchmarks will not be run.
```

Some notes:
- It is __not__ recommended to run the benchmarks inside the `R/` directory manually.
- It is best to be inside the current directory and run `bench.py`.
  This is because some utility files and variables get sourced prior to sourcing each benchmark file
  (see [File Structure](#file-structure)).
- If one changes any files not in the current directory, i.e. `R/` or `src/` files,
  from either the baseline or new version branch, one __must__ push the changes.
- For debugging purposes, if one wishes to run only one or a subset of the benchmarks,
  specify the `-i` flag with the benchmark name(s).

## File Structure

Every benchmark `.R` file must be inside the directory `R/`
and __must__ be prefixed with "bench" like `bench_foo.R`.
When running `bench.py`, it will automatically set the working directory to `R/`.
Then, it will first source `R/util.R`.
Afterwards, some variables are set for convenience
such as the default path to a data directory (`data/`)
where each benchmark can store any data (e.g. average benchmark times).
The benchmark of interest is then finally sourced.

`R/util.R` contains some useful functions that can be used across all benchmarks.
This makes it simpler to analyze the data as all data will be centralized
and `bench.py` will, by default, look in this path if analyzing flags are passed.

All benchmarks can output any number of data files containing any information
that its corresponding analyze script needs (see below).
This data is stored in a folder with the same benchmark name in `data/`.
For example, for the benchmark `R/bench_foo.R`, 
`bench.py` will create subdirectories 
`data/bench_foo/baseline` and `data/bench_foo/new_version`,
which will contain any data files that `R/bench_foo.R` generates
when run under the respective versions.

The directory `analyze` contains scripts that analyze the data.
All analyze scripts __must__ have the same name as its corresponding `R` benchmark script in `R/`.
For example, the analyze script for `bench_foo.R` must be named `bench_foo.py`.
They __must__ contain a function `analyze` that takes in three parameters:

```python
# baseline_data_path    path to baseline data (e.g. data/benchmark_name/baseline)
# new_version_data_path path to new version data (e.g. data/benchmark_name/new_version)
# fig_path              path to store figures (e.g. fig)
def analyze(baseline_data_path,
            new_version_data_path,
            fig_path):
    # some stuff...
```

When running `bench.py` with analyze flags, 
each analyze script is imported and the `analyze` function is run.

We recommend all future benchmarks to follow the same design for consistency.

An example command to run only the benchmark `bench_gaussian.R` and analyze (save figure),
assuming the user is in the current directory:
```
python bench.py -r https://github.com/trevorhastie/glmnet.git -b master -n your-new-branch -i gaussian --analyze
```

## Adding a Benchmark

As stated in [File Structure](#file-structure),
simply write a `bench_something.R` in `R/` that benchmarks a certain function,
and save the necessary output. 
See [bench_gaussian.R](R/bench_gaussian.R), [glmnet_util.R](R/glmnet_helper.R) as an example.
Then, write a `bench_something.py` in `analyze/` that reads the saved output,
plots, and saves the figure.
See [bench_gaussian.py](analyze/bench_gaussian.py), [glmnet_utils.py](analyze/glmnet_utils.py) as an example.

## Bug Reports

If there is a bug in `bench.py` or the benchmark scripts themselves, make an issue on GitHub.
