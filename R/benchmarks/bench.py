import os
import sys
import shutil
import argparse
import subprocess
import errno
import stat
import importlib
from colorama import Fore, Style, init

parser = argparse.ArgumentParser(description='Benchmark driver')
parser.add_argument('-d', nargs='?', default=os.getcwd(),
                    metavar='bench_dir',
                    help='Path to directory containing subdirectory "R/" \
                          that contains R benchmark scripts (default: current working directory).')
parser.add_argument('-r', nargs=1, required=True,
                    metavar='path_to_repo',
                    help='Path or URL to repository to benchmark.')
parser.add_argument('-b', nargs=1, required=True,
                    metavar='baseline_branch',
                    help='Branch or tag name for baseline.')
parser.add_argument('-n', nargs=1, required=True,
                    metavar='new_version_branch',
                    help='Branch or tag name for new version to benchmark \
                          against baseline.')
parser.add_argument('-c', action='store_const', const=True,
                    default=False,
                    help='If specified, cleans any files that the script \
                          produced from previous runs. \
                          This is necessary if the repository path/url \
                          changed from the previous run.')
parser.add_argument('-i', nargs='+',
                    metavar='include-subset',
                    help='A list of the subset of benchmarks to run. \
                          At least one name must be provided if the flag is passed. \
                          The names can either be the full name (e.g. bench_glmnet) \
                          or the keyword in the name (e.g. glmnet). ')
parser.add_argument('-e', nargs='+',
                    metavar='exclude-subset',
                    help='A list of the subset of benchmarks to exclude from running. \
                          At least one name must be provided if the flag is passed. \
                          The names can either be the full name (e.g. bench_glmnet) \
                          or the keyword in the name (e.g. glmnet).')
parser.add_argument('--analyze', action='store_const', const=True,
                    default=False,
                    help='Analyze also after running the benchmarks. \
                          The folder analyze/ containing analyze scripts must exist.')
parser.add_argument('--analyze-only', action='store_const', const=True,
                    default=False,
                    help='Analyze only and do not run the benchmarks. \
                          The folder analyze/ containing analyze scripts must exist. \
                          If --analyze is also passed, it is ignored, i.e. \
                          benchmarks will not be run.')

args = parser.parse_args()

# Global args
bench_path = args.d
repo_path = args.r[0]
baseline_branch = args.b[0]
new_version_branch = args.n[0]
bench_subset = args.i
bench_subset_exclude = args.e
do_analyze = args.analyze
do_analyze_only = args.analyze_only
baseline_name = "baseline"
new_version_name = "new_version"
do_clean = args.c
bench_prefix = "bench"
r_path = os.path.join(bench_path, "R")
analyze_path = os.path.join(bench_path, "analyze")
fig_path = os.path.join(bench_path, "fig")
tmp_name = ".tmp"
tmp_path = os.path.join(bench_path, tmp_name)
data_name = 'data'
data_path = os.path.join(bench_path, data_name)

# Pretty colors :D
class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = Fore.RED

color_dict = {
    "blue" : bcolors.BLUE, # currently unused
    "cyan" : bcolors.CYAN,
    "green" : bcolors.GREEN,
    "yellow" : bcolors.YELLOW,
    "red" : bcolors.RED
}

def pretty_print(line, color):
    print(color_dict[color] + line + Style.RESET_ALL)

def clear_dir(dir_path):

    # On Windows, we must change permissions to 0777 before removing file/folder.
    # This handler is passed to shutil.rmtree when we recursively empty out folders.
    def handleRemoveReadonly(func, path, exc):
        excvalue = exc[1]
        if func in (os.rmdir, os.unlink, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
            func(path)
        else:
            raise

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.chmod(file_path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=False, onerror=handleRemoveReadonly)
        except Exception as e:
            pretty_print('Failed to delete %s. Reason: %s' % (file_path, e), "red")

def cd(path, log=True):
    if log:
        pretty_print("[cd " + path + "]", "blue")
    os.chdir(path)

def cd_f_cd(f):
    def wrapper_cd_f_cd(path, log=False, *args, **kwargs):
        cwd = os.getcwd()
        cd(path, log)
        f(*args, **kwargs)
        cd(cwd, log)
    return wrapper_cd_f_cd

def prettify(f):
    def prettify_wrapper(*args,
                         init_msg=None,
                         init_msg_color=None,
                         exit_msg=None,
                         exit_msg_color=None,
                         do_success_msg=False,
                         **kwargs):
        if init_msg and init_msg_color:
            pretty_print(init_msg, init_msg_color)
        try:
            f(*args, **kwargs)
        except Exception as e:
            pretty_print("[Failed]", "red")
            raise e
        if exit_msg and exit_msg_color:
            pretty_print(exit_msg, exit_msg_color)
        if do_success_msg:
            pretty_print("[Success]", "green")
    return prettify_wrapper

@prettify
def run_cmd(cmd_split):
    print("> ", ' '.join(cmd_split), '\n')
    with subprocess.Popen(
        cmd_split,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='') # process line here
        for line in p.stderr:
            print(line, end='') # process line here

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)

def make_pref_suff(subset):
    return ["bench_" + f for f in subset] \
            if subset else \
            []

def is_included(filename, subset, subset_pref_suff):
    return (subset is None) or \
           (filename in subset) or \
           (filename in subset_pref_suff)

def is_excluded(filename, subset, subset_pref_suff):
    if subset is None:
        return False
    return (filename in subset) or \
           (filename in subset_pref_suff)

def bench():

    # make tmp directory if it doesn't exist
    if not os.path.exists(tmp_path):
        pretty_print("[mkdir -p {name}]".format(name=tmp_name), "blue")
        os.mkdir(tmp_path)

    # if we need to clean, empty the tmp directory
    if do_clean:
        pretty_print("[Cleaning " + tmp_name + " ...]", "cyan")
        clear_dir(tmp_path)

    # clone a repo to path given by dest
    # only clone if dest (baseline/new_version) doesn't exist
    # Note: currently it does not check if the repo was also the same,
    # so if a new repo path is given from a previous run, it will essentially be ignored.
    # This shouldn't be an issue for our use-case since it will always be glmnet.
    def clone(repo, dest):
        if os.path.exists(dest):
            return
        cmd = "git clone " + repo + " " + dest
        run_cmd(cmd.split())

    baseline_path = os.path.join(tmp_path, baseline_name)
    new_version_path = os.path.join(tmp_path, new_version_name)

    clone(repo_path, baseline_path)
    clone(repo_path, new_version_path)

    # set branch in each clone
    @cd_f_cd
    def set_branch(branch_name):
        cmd = "git fetch --all"
        run_cmd(cmd.split())
        cmd = "git checkout " + branch_name
        run_cmd(cmd.split())
        try:
            cmd = "git pull"
            run_cmd(cmd.split())
        except:
            pass

    set_branch(baseline_path, True, baseline_branch)
    set_branch(new_version_path, True, new_version_branch)

    # R/ directory must be in current working dir
    if not os.path.exists(r_path):
        raise Exception("The directory R/ must exist inside the benchmark directory: " +
                        bench_path)

    @cd_f_cd
    def run_benchmark(r_path, version_name):

        bench_subset_pref_suff = make_pref_suff(bench_subset)
        bench_subset_exclude_pref_suff = make_pref_suff(bench_subset_exclude)

        r_setup_code = \
        '''
        #!/usr/bin/env Rscript
        library(devtools);
        library(pkgbuild);
        pkgbuild::compile_dll(debug=F);
        load_all();
        setwd(\'{r_path}\')

        # source utility functions
        source(\'util.R\')

        # set some helper variables

        # path to benchmark specific data directory.
        # can assume that directory exists.
        data.path <- \'{data_path}\'

        # source benchmark
        source(\'{r_file}\')

        warnings()
        '''.strip()

        # go through all the .R files in R/ that start with "bench"
        for filename in os.listdir(r_path):
            filename_base = os.path.splitext(filename)[0]

            if not filename.endswith(".R") or \
               not filename.startswith("bench") or \
               not is_included(filename_base,
                               bench_subset,
                               bench_subset_pref_suff) or \
               is_excluded(filename_base,
                           bench_subset_exclude,
                           bench_subset_exclude_pref_suff):
                pretty_print("[Skipping " + filename + "...]", "yellow")
                continue

            pretty_print("[Processing " + filename + "...]", "cyan")

            # e.g. data_path/bench_something
            data_bench_path = \
                os.path.join(data_path,
                             filename_base)
            # e.g. data_path/bench_something/baseline
            data_bench_version_path = \
                os.path.join(data_bench_path,
                             version_name)

            # make benchmark specific data directory if it doesn't exist
            if not os.path.exists(data_path):
                pretty_print("[mkdir -p {path}]".format(path=data_path), "blue")
                os.mkdir(data_path)
            if not os.path.exists(data_bench_path):
                pretty_print("[mkdir -p {path}]".format(path=data_bench_path), "blue")
                os.mkdir(data_bench_path)
            if not os.path.exists(data_bench_version_path):
                pretty_print("[mkdir -p {path}]".format(path=data_bench_version_path), "blue")
                os.mkdir(data_bench_version_path)

            # clear the benchmark and version-specific data directory
            clear_dir(data_bench_version_path)

            # run a single benchmark file
            # note that we have to format the paths
            # to escape the backslash when passing to R.
            r_file = os.path.join(r_path, filename)
            r_driver_code = r_setup_code.format(
                r_path=r_path.replace('\\', '\\\\'),
                data_path=data_bench_version_path.replace('\\', '\\\\'),
                r_file=r_file.replace('\\', '\\\\')
            )
            cmd_split = ["Rscript", "-e", r_driver_code]
            run_cmd(cmd_split,
                    exit_msg="[Done: {bench}]".format(bench=filename_base),
                    exit_msg_color="green")

    @cd_f_cd
    def run_analyze(analyze_path):

        bench_subset_pref_suff = make_pref_suff(bench_subset)
        bench_subset_exclude_pref_suff = make_pref_suff(bench_subset_exclude)

        if not os.path.exists(fig_path):
            pretty_print("[mkdir -p {path}]".format(path=fig_path), "blue")
            os.mkdir(fig_path)

        # add analyze path to PATH temporarily since the scripts
        # may import helper modules in the same directory
        sys.path.append(analyze_path)

        for filename in os.listdir(analyze_path):
            filename_base = os.path.splitext(filename)[0]

            if not filename.endswith(".py") or \
               not filename.startswith("bench") or \
               not is_included(filename_base,
                               bench_subset,
                               bench_subset_pref_suff) or \
               is_excluded(filename_base,
                           bench_subset_exclude,
                           bench_subset_exclude_pref_suff):
                pretty_print("[Skipping " + filename + "...]", "yellow")
                continue

            pretty_print("[Processing " + filename + "...]", "cyan")

            # e.g. data_path/bench_something/baseline
            baseline_data_path = \
                os.path.join(data_path, filename_base, baseline_name)
            new_version_data_path = \
                os.path.join(data_path, filename_base, new_version_name)

            spec = importlib.util.spec_from_file_location(
                "bench",
                os.path.join(os.getcwd(), filename) )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            mod.analyze(
                baseline_data_path,
                new_version_data_path,
                fig_path)

            pretty_print("[Done: {bench}]".format(bench=filename_base),
                         "green")

    if not do_analyze_only:
        pretty_print("[Running benchmarks for baseline...]", "cyan")
        run_benchmark(baseline_path, True, r_path, baseline_name)
        pretty_print("[Done]", "green")
        pretty_print("[Running benchmarks for new version...]", "cyan")
        run_benchmark(new_version_path, True, r_path, new_version_name)
        pretty_print("[Done]", "green")

    if do_analyze or do_analyze_only:
        pretty_print("[Running analyze scripts...]", "cyan")
        run_analyze(analyze_path, True, analyze_path)
        pretty_print("[Done]", "green")

    return

if __name__ == '__main__':
    init() # colorama initialization
    bench()
