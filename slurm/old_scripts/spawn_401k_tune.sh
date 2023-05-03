# variables 
EXPR_ID="401k_tune_dents_temp_moar"

katas=("0.0" "1.0")
temps=("1000.0" "750.0" "500.0" "250.0" "100.0" "75.0" "50.0" "25.0" "1.0")

for AGENT_ONE_ID in ${katas[@]}
do
    for AGENT_TWO_ID in ${temps[@]}
    do
        OUTDIR=slurm_output/${EXPR_ID}
        mkdir -p $OUTDIR
        OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
        sbatch --job-name=t:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
    done
done