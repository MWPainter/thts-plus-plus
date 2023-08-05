# variables 
EXPR_ID="005_dents_hps"

temps=("0.05" "0.01" "0.005" "0.001" "0.0005")

for AGENT_ONE_ID in ${temps[@]}
do
    for AGENT_TWO_ID in ${temps[@]}
    do
        if [[ $AGENT_ONE_ID != $AGENT_TWO_ID ]]
        then
            OUTDIR=slurm_output/${EXPR_ID}
            mkdir -p $OUTDIR
            OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
            sbatch --job-name=5:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
        fi
    done
done