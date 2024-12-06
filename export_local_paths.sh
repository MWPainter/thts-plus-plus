# Need to specify where the python libraries are on each machine
anaconda_env_lib_dir="/home/michael/anaconda3/envs/thts3.12/lib"

dirname=$(readlink -f -- $(dirname "${BASH_SOURCE[0]}"))

export PYTHONPATH=$dirname:$dirname/py:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname/external/qhull/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname/external/clp/dist/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$anaconda_env_lib_dir

