# variables 
EXPR_ID="001_komi_9x9"
AGENT_ONE_ID="kata"
AGENT_TWO_ID="kata"

komis=("4.5" "5.5" "6.5" "7.5" "8.5" "9.5")

for KOMI in ${komis[@]}
do
        OUTDIR=slurm_output/${EXPR_ID}
        mkdir -p $OUTDIR
        OUTFILE=slurm_output/${EXPR_ID}/9x9_kata_vs_kata_komi_${KOMI}.out
        sbatch --job-name=komi_${KOMI} --output=$OUTFILE --export=EXPR_ID=$EXPR_ID,AGENT_ONE_ID=$KOMI,AGENT_TWO_ID=$KOMI run_go.slurm
done