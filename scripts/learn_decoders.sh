for encoding in baseline dissent.books8.epoch9 fairseq.wmt14.en-fr.fconv imdbsentiment infersent.allnli order-embeddings skipthought bert.base transformer.roc elmo.2x4096; do
    python learn_decoder.py data/stimuli_384sentences.txt encodings/384sentences.${encoding}.npy data \
        --encoding_project 256 --out_prefix perf.384sentences.${encoding} &
done

