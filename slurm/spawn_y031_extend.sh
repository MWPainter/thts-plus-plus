# variables 
EXPR_ID="y031_dp_round_robin"

agent_ids_new=("rents" "tents")
agent_ids_old=("kata" "ments" "dents" "est")

for AGENT_ONE_ID in ${agent_ids_new[@]}
do
    for AGENT_TWO_ID in ${agent_ids_old[@]}
    do
        if [[ $AGENT_ONE_ID != $AGENT_TWO_ID ]]
        then
            OUTDIR=slurm_output/${EXPR_ID}
            mkdir -p $OUTDIR
            OUTFILE=slurm_output/${EXPR_ID}/19x19_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
            sbatch --job-name=y31:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
        fi
    done
done

for AGENT_ONE_ID in ${agent_ids_old[@]}
do
    for AGENT_TWO_ID in ${agent_ids_new[@]}
    do
        if [[ $AGENT_ONE_ID != $AGENT_TWO_ID ]]
        then
            OUTDIR=slurm_output/${EXPR_ID}
            mkdir -p $OUTDIR
            OUTFILE=slurm_output/${EXPR_ID}/19x19_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
            sbatch --job-name=y31:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
        fi
    done
done

for AGENT_ONE_ID in ${agent_ids_new[@]}
do
    for AGENT_TWO_ID in ${agent_ids_new[@]}
    do
        if [[ $AGENT_ONE_ID != $AGENT_TWO_ID ]]
        then
            OUTDIR=slurm_output/${EXPR_ID}
            mkdir -p $OUTDIR
            OUTFILE=slurm_output/${EXPR_ID}/19x19_${AGENT_ONE_ID}_vs_${AGENT_TWO_ID}.out
            sbatch --job-name=y31:${AGENT_ONE_ID}vs${AGENT_TWO_ID} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$AGENT_ONE_ID,AGENT_TWO_ID=$AGENT_TWO_ID run_go.slurm
        fi
    done
done
