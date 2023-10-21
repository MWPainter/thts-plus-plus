# variables 
EXPR_ID="2k1_tune_init_decayed_temp_dbdents"

katas=("0.0" "1.0")
# temps=("1.0" "0.3" "0.03" "0.003" "0.0003")
temps=("100.0" "30.0" "10.0" "3.0" "1.0")

for AGENT_ONE_ID in ${katas[@]}
do
    for AGENT_TWO_ID in ${temps[@]}
    do
        OUTDIR=slurm_output/${EXPR_ID}
        mkdir -p $OUTDIR
        OUTFILE=slurm_output/${EXPR_ID}/9x9_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
        sbatch --job-name=2k1:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
    done
done