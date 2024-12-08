# variables 
EXPR_ID=$1
OUTDIR=slurm_output/${EXPR_ID}
mkdir -p $OUTDIR
OUTFILE=${OUTDIR}/$(date +%s).out
sbatch --job-name=${EXPR_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID run_hp_opt.slurm