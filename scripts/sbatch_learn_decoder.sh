#!/usr/bin/bash

#SBATCH --mem=16384
#SBATCH -t 1-0
#SBATCH -c 1
#SBATCH --qos=cpl

source /etc/profile.d/modules.sh
source ~/.profile
source activate decoding
echo "Learning decoder with encoding $encoding"
python learn_decoder.py data/stimuli_384sentences.txt data \
    encodings/384sentences.${encoding}.npy \
    --encoding_project 256 \
    --out_prefix perf.384sentences.${encoding}
