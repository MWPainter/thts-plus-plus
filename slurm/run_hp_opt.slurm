#!/bin/bash

# set number of node, processes (per node), and CPUs (per process)
# N.B. I think process = could be on a different node, so is for distributed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# set max wallclock time <days>-<hours>:<mins>:<secs>
#SBATCH --time=0-12:00:00

# # set number of GPUs
# # SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=BEGIN,END,FAIL

# send mail to this address
#SBATCH --mail-user=mpainter@robots.ox.ac.uk

# Use a small partition
#SBATCH --partition=short

# As we're comparing performance w.r.t time, want consistent CPU speeds
# Following https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html 
# Want htc-c[005-046], as they all have same CPU + clock speed
# Also 001-004 might be ok, but they're not on the specs list
#SBATCH --exclude=htc-c001,htc-c002,htc-c003,htc-c004,htc-c047
## htc-g[001-052] - should be avoided because we dont want to use gpu
## if we get allocated a htc-g at some point we'll cancel and type out a list of 52 nodes -.-


HOME_DIR="/home/pemb5587"
CODE_DIR="/home/pemb5587/thts-plus-plus"

echo Running experiment ${EXPR_ID}


cd $CODE_DIR                            # Get to code
source setup_arc_shell.sh               # Load conda env and modules 
source export_local_paths_arc.sh        # Setup $PYTHONPATH and $LD_LIBRARY_PATH environment variables 
# ipcrm -v -a                           # Make sure shared memory clear
./moexpr opt $EXPR_ID                   # Run