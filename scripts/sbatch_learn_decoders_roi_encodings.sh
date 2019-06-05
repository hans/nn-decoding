for encoding in QQP MNLI SST SQuAD LM_scrambled LM_pos 
do
    echo $encoding
    ./sbatch_learn_decoders_roi.sh $encoding &
done
