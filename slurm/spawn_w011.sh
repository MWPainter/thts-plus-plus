# variables 
EXPR_ID="w011_bts_tune_temp_most_visited_sqrt_decay"

temps=("3.0" "1.0" "0.3" "0.1" "0.03")

for AGENT_ONE_ID in ${temps[@]}
do
    for AGENT_TWO_ID in ${temps[@]}
    do
        if [[ $AGENT_ONE_ID != $AGENT_TWO_ID ]]
        then
            OUTDIR=slurm_output/${EXPR_ID}
            mkdir -p $OUTDIR
            OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
            sbatch --job-name=w11:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
        fi
    done
done