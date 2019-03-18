#!/usr/bin/bash

#SBATCH --mem=8G
#SBATCH -t 0-5
#SBATCH -c 3
#SBATCH --qos=cpl
#SBATCH -a 5-250:5%1

source /etc/profile.d/modules.sh
source ~/.profile
source activate decoding

ENCODING_NAME="${encoding}-${SLURM_ARRAY_TASK_ID}"

echo "Learning decoder with encoding $ENCODING_NAME"

for subject_dir in data/brains/*; do
    subject=`basename "$subject_dir"`
    python src/learn_decoder.py data/sentences/stimuli_384sentences.txt ${subject_dir} \
        models/bert/${ENCODING_NAME}.npy \
        --encoding_project 256 \
        --image_project 2048 \
        --n_jobs ${SLURM_JOB_CPUS_PER_NODE} \
        --n_folds 8 \
        --out_prefix models/decoders/${ENCODING_NAME}-${subject}
done
