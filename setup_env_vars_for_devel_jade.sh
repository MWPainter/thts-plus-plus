this_filename=$( realpath "$0" )
dirname=$( dirname "$this_filename" )
export PYTHONPATH=$dirname:$dirname/py:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$dirname/external/qhull/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts3.12/lib

