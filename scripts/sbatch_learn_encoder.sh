#!/usr/bin/bash

#SBATCH --mem=16G
#SBATCH -t 1-0
#SBATCH -c 4
#SBATCH --qos=cpl
#SBATCH -a 0-6%4

source /etc/profile.d/modules.sh
source ~/.profile
source activate decoding

LAYERS=(0 2 5 8 9 10 11)
layer=${LAYERS[$SLURM_ARRAY_TASK_ID]}
ENCODING_NAME="${encoding}-layer$layer"

echo "Learning decoder with encoding $ENCODING_NAME"

for subject_dir in data/brains/*; do
    subject=`basename "$subject_dir"`
    echo $subject
    python src/learn_decoder.py data/sentences/stimuli_384sentences.txt ${subject_dir} \
        models/bert/${ENCODING_NAME}.npy \
        -e \
        --encoding_project 256 \
        --image_downsample 2 \
        --n_jobs ${SLURM_JOB_CPUS_PER_NODE} \
        --n_folds 8 \
        --out_prefix models/encoders/${ENCODING_NAME}-${subject}
done
