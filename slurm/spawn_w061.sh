# variables 
EXPR_ID="w061_tents_tune_temp_most_visit"

# temps=("3.0" "1.0" "0.3" "0.1" "0.03")
# temps=("300.0" "100.0" "30.0" "10.0" "3.0")
temps=("10000.0" "3000.0" "1000.0" "300.0" "100.0")

for AGENT_ONE_ID in ${temps[@]}
do
    for AGENT_TWO_ID in ${temps[@]}
    do
        if [[ $AGENT_ONE_ID != $AGENT_TWO_ID ]]
        then
            OUTDIR=slurm_output/${EXPR_ID}
            mkdir -p $OUTDIR
            OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
            sbatch --job-name=w61:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
        fi
    done
done