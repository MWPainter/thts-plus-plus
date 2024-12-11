# variables 
EXPR_ID="y030_ar_round_robin"

num_fixup=2   # REMEMBER TO SET THIS BECAUSE I SUCK AT BASH LOOPS
agent_one_ids=("rents" "rents" )
agent_two_ids=("kata" "est")

i=0
until [ $i -ge $num_fixup ]
do
    AGENT_ONE_ID=${agent_one_ids[$i]}
    AGENT_TWO_ID=${agent_two_ids[$i]}

    OUTDIR=slurm_output/${EXPR_ID}
    mkdir -p $OUTDIR
    OUTFILE=slurm_output/${EXPR_ID}/19x19_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
    sbatch --job-name=y30:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm

    ((i++))
done