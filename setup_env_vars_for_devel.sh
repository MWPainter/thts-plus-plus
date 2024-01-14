dirname=$(readlink -f -- $(dirname "${BASH_SOURCE[0]}"))
export PYTHONPATH=$dirname:$dirname/py:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname/external/qhull/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/michael/anaconda3/envs/thts3.12/lib

