# variables 
EXPR_ID="013_dirichlet_noise"

katas=("0.0" "1.0")

for AGENT_ONE_ID in ${katas[@]}
do
    OUTDIR=slurm_output/${EXPR_ID}
    mkdir -p $OUTDIR
    OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
    sbatch --job-name=13:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_ONE_ID run_go.slurm
done