#!/usr/bin/env nextflow

// baseDir as prepared by nextflow references a particular fs share, which is
// not good
omBaseDir = "/om2/user/jgauthie/scratch/nn-decoding"
params.bert_dir = "~/om2/others/bert"
params.bert_base_model = "uncased_L-12_H-768_A-12"
params.glue_base_dir = "~/om2/data/GLUE"
params.squad_dir = "/om/data/public/jgauthie/squad-2.0"

params.brain_data_path = "$omBaseDir/data/brains"
params.sentences_path = "$omBaseDir/data/sentences/stimuli_384sentences.txt"

// Finetune parameters
params.finetune_steps = 250
params.finetune_checkpoint_steps = 5
params.finetune_learning_rate = "2e-5"
params.finetune_squad_learning_rate = "3e-5"
// CLI params shared across GLUE and SQuAD tasks
bert_base_dir = [params.bert_dir, params.bert_base_model].join("/")
finetune_cli_params = """--do_train=true --do_eval=true \
    --bert_config_file=${bert_base_dir}/bert_config.json \
    --vocab_file=${bert_base_dir}/vocab.txt \
    --init_checkpoint=${bert_base_dir}/model.ckpt \
    --num_train_steps=${params.finetune_steps} \
    --save_checkpoint_steps=${params.finetune_checkpoint_steps} \
    --output_dir ."""

// Encoding extraction parameters.
params.extract_encoding_layers = "-1"

// Decoder learning parameters
params.decoder_projection = 256
params.brain_projection = 8192
params.decoder_n_jobs = 5
params.decoder_n_folds = 8

params.outdir = "output"
params.publishDirPrefix = "${workflow.runName}"

/////////

glue_tasks = Channel.from("MNLI", "SST", "QQP")
brain_images = Channel.fromPath("${params.brain_data_path}/*", type: "dir")

process finetuneGlue {
    label "om_gpu_tf"
    publishDir "${params.outdir}/${params.publishDirPrefix}/bert"

    input:
    val glue_task from glue_tasks

    output:
    set glue_task, "model.ckpt-*" into model_ckpt_files_glue

    tag "$glue_task"

    """
#!/bin/bash
python ${params.bert_dir}/run_classifier.py --task_name=$glue_task \
    ${finetune_cli_params} \
    --data_dir=${params.glue_base_dir}/${glue_task} \
    --learning_rate ${params.finetune_learning_rate} \
    --max_seq_length 128 \
    --train_batch_size 32 \
    """
}

process finetuneSquad {
    label "om_gpu_tf"
    publishDir "${params.outdir}/${params.publishDirPrefix}/bert"

    output:
    set val("SQuAD"), "model.ckpt-*" into model_ckpt_files_squad

    tag "SQuAD"

    """
#!/bin/bash
python ${params.bert_dir}/run_squad.py \
    ${finetune_cli_params} \
    --train_file=${params.squad_dir}/train-v2.0.json \
    --predict_file=${params.squad_dir}/dev-v2.0.json \
    --data_dir=${params.squad_dir} \
    --max_seq_length 384 \
    --train_batch_size 12 \
    --doc_stride 128 \
    --learning_rate ${params.finetune_squad_learning_rate} \
    --version_2_with_negative=True
    """
}

model_ckpt_files_squad.into { squad_for_eval; squad_for_extraction }

// SQuAD training does not support online evaluation -- run separately on the
// saved checkpoints
// Group SQuAD checkpoints based on their prefix.
squad_for_eval.flatMap { ckpt_id -> ckpt_id[1].collect {
    file -> tuple((file.name =~ /^model.ckpt-(\d+)/)[0][1], file)
} }.groupTuple().set { squad_eval_ckpts }
process evalSquad {
    label "om_gpu_tf"
    publishDir "${params.outdir}/${params.publishDirPrefix}/eval_squad"

    input:
    set ckpt_step, file(ckpt_files) from squad_eval_ckpts

    output:
    set val("step ${ckpt_step}"), file("predictions.json"), file("null_odds.json"), file("results.json") into squad_eval_results

    script:
    """
#!/bin/bash

# Output a dummy checkpoint metadata file.
echo "model_checkpoint_path: \"model.ckpt-${ckpt_step}\"" > checkpoint

# Run prediction.
python ${params.bert_dir}/run_squad.py --do_predict \
    --vocab_file=${bert_base_dir}/vocab.txt \
    --bert_config_file=${bert_base_dir}/bert_config.json \
    --init_checkpoint=model.ckpt-${ckpt_step} \
    --predict_file=${params.squad_dir}/dev-v2.0.json \
    --doc_stride 128 --version_2_with_negative=True \
    --predict_batch_size 32 \
    --output_dir .

# Evaluate using SQuAD tools.
python ${params.squad_dir}/evaluate-v2.0.py ${params.squad_dir}/dev-v2.0.json \
    predictions.json --na-prob-file null_odds.json > results.json
    """
}

model_ckpt_files_glue.concat(squad_for_extraction).set { model_ckpt_files }

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

    """
#!/bin/bash
python ${params.bert_dir}/extract_features.py \
    --input_file=${params.sentences_path} \
    --output_file=encodings.jsonl \
    --vocab_file=${bert_base_dir}/vocab.txt \
    --bert_config_file=${bert_base_dir}/bert_config.json \
    --init_checkpoint=model.ckpt-${ckpt_id[1]} \
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

process learnDecoder {
    label "om"
    publishDir "${params.outdir}/${params.publishDirPrefix}/decoders"
    clusterOptions "${baseClusterOptions} -c ${params.decoder_n_jobs}"
    memory "8 GB"
    cpus params.decoder_n_jobs

    input:
    set model_id, file(encoding), file(brain_dir) from encodings_brains

    output:
    file "${model_id[0]}-${model_id[1]}-*"

    tag "${model_id[0]}-${model_id[1]}"

    """
#!/bin/bash
source activate decoding
python ${omBaseDir}/src/learn_decoder.py ${params.sentences_path} \
    ${brain_dir} ${encoding} \
    --n_jobs ${params.decoder_n_jobs} \
    --n_folds ${params.decoder_n_folds} \
    --out_prefix "${model_id[0]}-${model_id[1]}-${brain_dir.name}" \
    --encoding_project ${params.decoder_projection} \
    --image_project ${params.brain_projection}
    """
}
