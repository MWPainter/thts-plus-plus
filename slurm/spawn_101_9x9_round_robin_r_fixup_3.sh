# variables 
EXPR_ID="101_round_robin_9x9"

num_fixup=10
agent_one_ids=("tents" "tents" "tents" "tents" "tents" "kata" "est" "dents" "rents" "ments")
agent_two_ids=("kata" "est" "dents" "rents" "ments" "tents" "tents" "tents" "tents" "tents")

i=0
until [ $i -ge $num_fixup ]
do
    AGENT_ONE_ID=${agent_one_ids[$i]}
    AGENT_TWO_ID=${agent_two_ids[$i]}

    OUTDIR=slurm_output/${EXPR_ID}
    mkdir -p $OUTDIR
    OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
    sbatch --job-name=${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm

    ((i++))
done