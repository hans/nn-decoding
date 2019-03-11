#!/usr/bin/env nextflow

params.bert_dir = "~/om2/others/bert"
params.bert_base_model = "uncased_L-12_H-768_A-12"
params.glue_base_dir = "~/om2/data/GLUE"

params.brain_data_path = "$baseDir/data/brains"
params.sentences_path = "$baseDir/data/stimuli_384sentences.txt"

// Finetune parameters
params.finetune_steps = 250
params.finetune_checkpoint_steps = 5
params.finetune_learning_rate = "2e-5"

// Encoding extraction parameters.
params.extract_encoding_layers = "-1"

// Decoder learning parameters
params.decoder_projection = 256
params.decoder_n_jobs = 5

params.outdir = "output"
params.publishDirPrefix = "${workflow.runName}"

/////////

bert_tasks = Channel.from("MNLI", "SST", "QQP", "SQuAD")
brain_images = Channel.fromPath("${params.brain_data_path}/*", type: "dir")

process finetune {
    label "om_gpu_tf"
    publishDir "${params.outdir}/${params.publishDirPrefix}/bert"

    input:
    val glue_task from bert_tasks

    output:
    set glue_task, "model.ckpt-*" into model_ckpt_files

    tag "$glue_task"

    bert_base_dir = [params.bert_dir, params.bert_base_model].join("/")

    """
#!/bin/bash
python ${params.bert_dir}/run_classifier.py --task_name=$glue_task \
    --do_train=true --do_eval=true \
    --data_dir=${params.glue_base_dir}/${glue_task} \
    --vocab_file ${bert_base_dir}/vocab.txt \
    --bert_config_file ${bert_base_dir}/bert_config.json \
    --init_checkpoint ${bert_base_dir}/bert_model.ckpt \
    --num_train_steps ${params.finetune_steps} \
    --save_checkpoint_steps ${params.finetune_checkpoint_steps} \\
    --learning_rate ${params.finetune_learning_rate} \\
    --max_seq_length 128 \
    --train_batch_size 32 \
    --output_dir .
    """
}

// Group model checkpoints based on their prefix.
model_ckpt_files
    .flatMap { ckpt_id -> ckpt_id[1].collect {
        file -> tuple(tuple(ckpt_id[0], (file.name =~ /^model.ckpt-(\d+)/)[0][1]),
                      file) } }
    .groupTuple()
    .set { model_ckpts }

process extractEncoding {
    label "om_gpu_tf"

    input:
    set ckpt_id, file(ckpt_files) from model_ckpts

    output:
    set ckpt_id, "encodings.jsonl" into encodings_jsonl

    tag "${ckpt_id[0]}-${ckpt_id[1]}"

    bert_base_dir = [params.bert_dir, params.bert_base_model].join("/")

    """
#!/bin/bash
python ${params.bert_dir}/extract_features.py \
    --input_file=${params.sentences_path} \
    --output_file=encodings.jsonl \
    --vocab_file=${bert_base_dir}/vocab.txt \
    --bert_config_file=${bert_base_dir}/bert_config.json \
    --init_checkpoint=${ckpt_id[0]} \
    --layers="${params.extract_encoding_layers}" \
    --max_seq_length=128 \
    --batch_size=64
    """
}

process convertEncoding {
    label "om"
    publishDir "${params.outdir}/${params.publishDirPrefix}/encodings"

    input:
    set model_id, file(encoding_jsonl) from encodings_jsonl

    output:
    set model_id, "${model_id[0]}-${model_id[1]}.npy" into encodings

    tag "${model_id[0]}-${model_id[1]}"

    """
#!/usr/bin/bash
source activate decoding
python ${params.bert_dir}/process_encodings.py \
    -i ${encoding_jsonl} \
    -l ${params.extract_encoding_layers} \
    -o ${model_id[0]}-${model_id[1]}.npy
    """
}

encodings.combine(brain_images).set { encodings_brains }

process learnDecoder {
    label "om_big"
    publishDir "${params.outdir}/${params.publishDirPrefix}/decoders"

    input:
    set model_id, file(encoding), file(brain_dir) from encodings_brains

    output:
    file "${model_id[0]}-${model_id[1]}-${brain_dir.name}*"

    tag "${model_id[0]}-${model_id[1]}-${brain_dir.name}"

    """
#!/bin/bash
source activate decoding
python src/learn_decoder.py ${params.sentences_path} ${brain_dir} ${encoding} \
    --out_prefix "${model_id[0]}-${model_id[1]}-${brain_dir.name}" \
    --encoding_project ${decoder_projection}
    """
}
