# Need to specify where the python libraries are on each machine
anaconda_env_lib_dir="/data/engs-goals/pemb5587/anaconda3/envs/thts++mo/lib"

dirname=$(readlink -f -- $(dirname "${BASH_SOURCE[0]}"))

# python path
export PYTHONPATH=$dirname/main/envs:$PYTHONPATH        # so python can find envs for experiments
export PYTHONPATH=$dirname/main/envs/dst:$PYTHONPATH        # so python can find envs for experiments
export PYTHONPATH=$dirname/py:$PYTHONPATH               # so python can find python code in thts-plus-plus/py dir
export PYTHONPATH=$dirname:$PYTHONPATH                  # so python can find python code in thts-plus-plus dir

# linker paths (so can find libraries used in thts++ programs at runtime)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname/external/qhull/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname/external/clp/dist/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$anaconda_env_lib_dir

