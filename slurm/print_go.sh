# variables 
JOB_ID="print"
OUTFILE=slurm_output/${JOB_ID}.out

sbatch --job-name=${JOB_ID} --output=$OUTFILE print_go.slurm
