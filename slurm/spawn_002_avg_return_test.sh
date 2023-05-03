# variables 
EXPR_ID="002_avg_return_test"

katas=("0.0" "1.0")
agent_ids=("ments" "db-ments" "dents" "est" "rents")

for AGENT_ONE_ID in ${katas[@]}
do
    for AGENT_TWO_ID in ${agent_ids[@]}
    do
        OUTDIR=slurm_output/${EXPR_ID}
        mkdir -p $OUTDIR
        OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
        sbatch --job-name=t:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
    done
done