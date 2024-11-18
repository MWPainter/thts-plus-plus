#!/bin/bash

# Set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=NONE

# send mail to this address
#SBATCH --mail-user=mpainter@robots.ox.ac.uk

# Use a small partition
#SBATCH --partition=small



HOME_DIR="/jmain02/home/J2AD008/wga37/mmp10-wga37"
CODE_DIR="/jmain02/home/J2AD008/wga37/mmp10-wga37/thts-plus-plus"

echo Running experiment ${EXPR_ID}

cd $CODE_DIR
source setup_jade_devel_shell.sh
source setup_env_vars_for_devel_jade.sh
./moexpr opt $EXPR_ID