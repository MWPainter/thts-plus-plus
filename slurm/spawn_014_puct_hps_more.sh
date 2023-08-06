# variables 
EXPR_ID="014_puct_bias_hps"

num_fixup=12
agent_one_ids=("100.0" "50.0" "20.0" "10.0" "5.0" "2.0" "-1.0" "-1.0" "-1.0" "-1.0" "-1.0" "-1.0")
agent_two_ids=("-1.0" "-1.0" "-1.0" "-1.0" "-1.0" "-1.0" "2.0" "5.0" "10.0" "20.0" "50.0" "100.0")

i=0
until [ $i -ge $num_fixup ]
do
    AGENT_ONE_ID=${agent_one_ids[$i]}
    AGENT_TWO_ID=${agent_two_ids[$i]}

    OUTDIR=slurm_output/${EXPR_ID}
    mkdir -p $OUTDIR
    OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
    sbatch --job-name=14:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm

    ((i++))
done
