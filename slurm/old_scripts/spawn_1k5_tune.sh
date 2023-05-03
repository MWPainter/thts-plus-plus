# variables 
EXPR_ID="1k5_tune_prior_coeff"

# temps=("0.03" "0.01" "0.003" "0.001" "0.0003" "0.00003")
katas=("0.0" "1.0")
temps=("0.1" "0.3" "0.5" "0.7" "0.9")

for AGENT_ONE_ID in ${katas[@]}
do
    for AGENT_TWO_ID in ${temps[@]}
    do
        OUTDIR=slurm_output/${EXPR_ID}
        mkdir -p $OUTDIR
        OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
        sbatch --job-name=1k5:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
    done
done