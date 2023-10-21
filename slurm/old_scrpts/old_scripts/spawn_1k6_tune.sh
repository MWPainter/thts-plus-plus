# variables 
EXPR_ID="1k6_tune_decayed_temp_visits_scale_dents"

# temps=("0.03" "0.01" "0.003" "0.001" "0.0003" "0.00003")
katas=("0.0" "1.0")
temps=("0.1" "0.03" "0.01" "0.003" "0.001")

for AGENT_ONE_ID in ${katas[@]}
do
    for AGENT_TWO_ID in ${temps[@]}
    do
        OUTDIR=slurm_output/${EXPR_ID}
        mkdir -p $OUTDIR
        OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
        sbatch --job-name=1k6:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
    done
done