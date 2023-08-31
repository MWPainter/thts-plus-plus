# variables 
EXPR_ID="y02_round_robin_with_alias"

# agent_ids=("kata" "ments" "dents" "est" "rents" "tents")
agent_ids=("kata" "ments" "est" "rents")

for AGENT_ONE_ID in ${agent_ids[@]}
do
    for AGENT_TWO_ID in ${agent_ids[@]}
    do
        if [[ $AGENT_ONE_ID != $AGENT_TWO_ID ]]
        then
            OUTDIR=slurm_output/${EXPR_ID}
            mkdir -p $OUTDIR
            OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
            sbatch --job-name=${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
        fi
    done
done