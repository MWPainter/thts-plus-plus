# variables 
EXPR_ID="002_9x9_round_robin"
AGENT_ONE_ID="kata"
AGENT_TWO_ID="ments"

OUTDIR=slurm_output/${EXPR_ID}
mkdir -p $OUTDIR
OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
sbatch --job-name=${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm