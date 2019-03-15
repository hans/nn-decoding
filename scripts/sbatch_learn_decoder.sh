#!/usr/bin/bash

#SBATCH --mem=4G
#SBATCH -t 0-1
#SBATCH -c 3
#SBATCH --qos=cpl
#SBATCH -a 5-250:5%1

source /etc/profile.d/modules.sh
source ~/.profile
source activate decoding
echo "Learning decoder with encoding $encoding"
python src/learn_decoder.py data/sentences/stimuli_384sentences.txt data/brains/${brain} \
    models/bert/${encoding}-${SLURM_ARRAY_TASK_ID}.npy \
    --encoding_project 256 \
    --n_jobs ${SLURM_JOB_CPUS_PER_NODE} \
    --n_folds 8 \
    --out_prefix models/decoders/${encoding}
