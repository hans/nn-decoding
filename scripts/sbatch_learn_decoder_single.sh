#!/usr/bin/bash

#SBATCH --mem=8G
#SBATCH -t 0-5
#SBATCH -c 3
#SBATCH --qos=cpl

source /etc/profile.d/modules.sh
source ~/.profile
source activate decoding

echo "Learning decoder with encoding $ENCODING_NAME"

for subject_dir in data/brains/*; do
    subject=`basename "$subject_dir"`
    python src/learn_decoder.py data/sentences/stimuli_384sentences.txt ${subject_dir} \
        models/bert/${ENCODING_NAME}.npy \
        --encoding_project 256 \
        --image_project 256 \
        --n_jobs ${SLURM_JOB_CPUS_PER_NODE} \
        --n_folds 8 \
        --out_prefix models/decoders/${ENCODING_NAME}-${subject}
done
