#!/usr/bin/bash

#SBATCH --mem=16384
#SBATCH -t 1-0
#SBATCH -c 1
#SBATCH --qos=cpl
#SBATCH --array=5-250:5%4

# Learn decoders over a whole training trajectory.

source /etc/profile.d/modules.sh
source ~/.profile
source activate decoding

step=$SLURM_ARRAY_TASK_ID

echo "Learning decoder with encoding $encoding, step $step"
python learn_decoder.py data/stimuli_384sentences.txt data \
    encodings/384sentences.${encoding}-${step}.npy \
    --encoding_project 256 \
    --out_prefix perf.384sentences.${encoding}-${step}
