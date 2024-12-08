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


HOME_DIR="/home/pemb5587"
CODE_DIR="/home/pemb5587/thts-plus-plus"

echo Running experiment ${EXPR_ID}

# Run code
cd $CODE_DIR                        # Get to code
source setup_remote_shell.sh        # Load modules and setup shell on remote
source export_local_paths.sh        # Setup $PYTHONPATH and $LD_LIBRARY_PATH environment variables 
# ipcrm -v -a                         # Make sure shared memory clear
./moexpr opt $EXPR_ID               # Run