#!/usr/bin/env nextflow

import org.yaml.snakeyaml.Yaml

params.bert_dir = "/om2/user/jgauthie/others/bert"
params.bert_base_model = "uncased_L-12_H-768_A-12"

// Finetune parameters
params.finetune_steps = 250
params.finetune_checkpoint_steps = 5
params.finetune_learning_rate = "2e-5"
params.finetune_squad_learning_rate = "3e-5"
// CLI params shared across GLUE and SQuAD tasks
// TODO make sure we can get rid of all references to bert_base_dir
// so that we can run containerized BERT
bert_base_dir = [params.bert_dir, "models", params.bert_base_model].join("/")
finetune_cli_params = """--do_train=true --do_eval=true \
    --bert_config_file=${bert_base_dir}/bert_config.json \
    --vocab_file=${bert_base_dir}/vocab.txt \
    --init_checkpoint=${bert_base_dir}/model.ckpt \
    --num_train_steps=${params.finetune_steps} \
    --save_checkpoint_steps=${params.finetune_checkpoint_steps} \
    --output_dir ."""

// Encoding extraction parameters.
params.extract_encoding_layers = "-1"
params.extract_encoding_cls = true

// Decoder learning parameters
params.decoder_projection = 256
params.brain_projection = 256
params.decoder_n_jobs = 5
params.decoder_n_folds = 8

// Structural probe parameters
params.structural_probe_layers = "11"
structural_probe_layers = params.structural_probe_layers.split(",")
params.structural_probe_spec = "structural-probes/spec.yaml"
structural_probe_spec = new Yaml().load((params.structural_probe_spec as File).text)

/////////

params.outdir = "output"

// TODO make container
params.bert_container = "library://jon/default/bert:small-gpu"
params.structural_probes_container = "library://jon/default/structural-probes:latest"
// TODO make container
params.decoding_container = "library://jon/default/nn-decoding:emnlp2019"

/////////

glue_tasks = Channel.from("MNLI", "SST", "QQP")
brain_images = Channel.fromPath([
    "https://www.dropbox.com/s/bdll04a2h4ou4xj/P01.tar?dl=1",
    "https://www.dropbox.com/s/wetd2gqljfbh8cg/M02.tar?dl=1",
    "https://www.dropbox.com/s/b7tvvkrhs5g3blc/M04.tar?dl=1",
    "https://www.dropbox.com/s/izwr74rxn637ilm/M07.tar?dl=1",
    "https://www.dropbox.com/s/3q6xhtmj611ibmo/M08.tar?dl=1",
    "https://www.dropbox.com/s/kv1wm2ovvejt9pg/M09.tar?dl=1",
    "https://www.dropbox.com/s/2h6kmootoruwz52/M14.tar?dl=1",
    "https://www.dropbox.com/s/u19wdpohr5pzohr/M15.tar?dl=1",
])

/**
 * Uncompress brain image data.
 */
process extractBrainData {
    label "small"
    publishDir "${params.outdir}/brains"

    input:
    file("*.tar*") from brain_images.collect()

    output:
    file("*", type: "dir") into brain_images_uncompressed

    """
#!/usr/bin/env bash
find . -name '*tar*' | while read -r path; do
    newpath="\${path%.*}"
    mv "\$path" "\$newpath"
    tar xf "\$newpath"
    rm "\$newpath"
done
    """
}

sentence_data = Channel.fromPath("https://www.dropbox.com/s/jtqnvzg3jz6dctq/stimuli_384sentences.txt?dl=1")
sentence_data.into { sentence_data_for_extraction; sentence_data_for_decoder }

/**
 * Fetch GLUE task data (except SQuAD).
 */
process fetchGLUEData {
    label "small"

    output:
    file("GLUE", type: "dir") as glue_data

    """
#!/usr/bin/env bash
# TODO download GLUE data
    """
}

/**
 * Fetch the SQuAD dataset.
 */
Channel.fromPath("TODO SQuAD URL").into { squad_data_for_train; squad_data_for_eval }

/**
 * Fine-tune and evaluate the BERT model on the GLUE datasets (except SQuAD).
 */
process finetuneGlue {
    label "gpu_large"
    container params.bert_container
    publishDir "${params.outdir}/bert"

    input:
    val(glue_task), file(glue_dir) from glue_tasks.combine(glue_data)

    output:
    set glue_task, "model.ckpt-*" into model_ckpt_files_glue

    tag "$glue_task"

    script:
    // TODO assert that glue_task exists in glue_dir

    """
#!/bin/bash
run_classifier.py --task_name=$glue_task \
    ${finetune_cli_params} \
    --data_dir=${glue_dir}/${glue_task} \
    --learning_rate ${params.finetune_learning_rate} \
    --max_seq_length 128 \
    --train_batch_size 32 \
    """
}

/**
 * Fine-tune the BERT model on the SQuAD dataset.
 */
process finetuneSquad {
    label "gpu_large"
    container params.bert_container
    publishDir "${params.outdir}/bert"

    input:
    file(squad_dir) from squad_data_for_train

    output:
    set val("SQuAD"), "model.ckpt-*" into model_ckpt_files_squad

    tag "SQuAD"

    """
#!/bin/bash
run_squad.py \
    ${finetune_cli_params} \
    --train_file=${squad_dir}/train-v2.0.json \
    --predict_file=${squad_dir}/dev-v2.0.json \
    --data_dir=${squad_dir} \
    --max_seq_length 384 \
    --train_batch_size 12 \
    --doc_stride 128 \
    --learning_rate ${params.finetune_squad_learning_rate} \
    --version_2_with_negative=True
    """
}

model_ckpt_files_squad.into { squad_for_eval; squad_for_extraction }

/**
 * Run evaluation for the SQuAD fine-tuned models.
 */
// Group SQuAD checkpoints based on their prefix.
squad_for_eval.flatMap { ckpt_id -> ckpt_id[1].collect {
    file -> tuple((file.name =~ /^model.ckpt-(\d+)/)[0][1], file)
} }.groupTuple().set { squad_eval_ckpts }
process evalSquad {
    label "gpu_medium"
    container params.bert_container
    publishDir "${params.outdir}/eval_squad"

    input:
    set ckpt_step, file(ckpt_files), file(squad_dir) \
        from squad_eval_ckpts.combine(squad_data_for_eval)

    output:
    set val("step ${ckpt_step}"), file("predictions.json"), file("null_odds.json"), file("results.json") into squad_eval_results

    script:
    """
#!/bin/bash

# Output a dummy checkpoint metadata file.
echo "model_checkpoint_path: \"model.ckpt-${ckpt_step}\"" > checkpoint

# Run prediction.
run_squad.py --do_predict \
    --vocab_file=${bert_base_dir}/vocab.txt \
    --bert_config_file=${bert_base_dir}/bert_config.json \
    --init_checkpoint=model.ckpt-${ckpt_step} \
    --predict_file=${squad_dir}/dev-v2.0.json \
    --doc_stride 128 --version_2_with_negative=True \
    --predict_batch_size 32 \
    --output_dir .

# Evaluate using SQuAD tools.
python ${squad_dir}/evaluate-v2.0.py ${squad_dir}/dev-v2.0.json \
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
    .into { model_ckpts_for_decoder; model_ckpts_for_sprobe }

/**
 * Extract .jsonl sentence encodings from each fine-tuned model.
 */
process extractEncoding {
    label "gpu_medium"
    container params.bert_container

    input:
    set run_id, file(ckpt_files), file(sentences) \
        from model_ckpts_for_decoder.combine(sentence_data_for_extraction)

    output:
    set run_id, "encodings*.jsonl" into encodings_jsonl

    tag "${run_id}"

    script:
    all_ckpts = ckpt_files.target.collect { (it.name =~ /^model.ckpt-(\d+)/)[0][1] }.unique()
    all_ckpts_str = all_ckpts.join(" ")

    """
#!/bin/bash

for ckpt in ${all_ckpts_str}; do
    extract_features.py \
        --input_file=${sentences} \
        --output_file=encodings-\$ckpt.jsonl \
        --vocab_file=${bert_base_dir}/vocab.txt \
        --bert_config_file=${bert_base_dir}/bert_config.json \
        --init_checkpoint=model.ckpt-\$ckpt \
        --layers="${params.extract_encoding_layers}" \
        --max_seq_length=128 \
        --batch_size=64
done
    """
}

// Expand jsonl encodings into individual identifier + jsonl files
encodings_jsonl.flatMap {
    els -> els[1].collect {
        f -> [[els[0], (f.name =~ /-(\d+).jsonl/)[0][1]].join("-"), f]
    }
}.set { encodings_jsonl_flat }

/**
 * Convert .jsonl encodings to easier-to-use numpy arrays, saved as .npy
 */
process convertEncoding {
    label "medium"
    container params.bert_container
    publishDir "${params.outdir}/encodings"

    input:
    set ckpt_id, file(encoding_jsonl) from encodings_jsonl_flat

    output:
    set ckpt_id, "*.npy" into encodings

    tag "${ckpt_id}"

    script:
    if (params.extract_encoding_cls) {
        modifier_flag = "-c"
    } else {
        modifier_flag = "-l ${params.extract_encoding_layers}"
    }

    """
#!/usr/bin/bash
process_encodings.py \
    -i ${encoding_jsonl} \
    ${modifier_flag} \
    -o ${ckpt_id}.npy
    """
}

encodings.combine(brain_images).set { encodings_brains }

/**
 * Learn regression models mapping between brain images and model encodings.
 */
process learnDecoder {
    label "medium"
    container params.decoding_container

    publishDir "${params.outdir}/decoders"
    cpus params.decoder_n_jobs

    input:
    set ckpt_id, file(encoding), file(brain_dir) from encodings_brains
    file(sentences) from sentence_data_for_decoder

    output:
    file "${ckpt_id}-*"

    tag "${ckpt_id}-${brain_dir.name}"

    script:
    """
#!/bin/bash
python src/learn_decoder.py ${sentences} \
    ${brain_dir} ${encoding} \
    --n_jobs ${params.decoder_n_jobs} \
    --n_folds ${params.decoder_n_folds} \
    --out_prefix "${ckpt_id}-${brain_dir.name}" \
    --encoding_project ${params.decoder_projection} \
    --image_project ${params.brain_projection}
    """
}

/**
 * Extract encodings for structural probe analysis (expects hdf5 format).
 */
process extractEncodingForStructuralProbe {
    label "gpu_medium"
    container params.bert_container

    input:
    set run_id, file(ckpt_files) from model_ckpts_for_sprobe

    output:
    set run_id, "encodings-*.hdf5" into encodings_sprobe

    tag "${run_id}"

    script:
    all_ckpts = ckpt_files.collect { (it.name =~ /^model.ckpt-(\d+)/)[0][1] }.unique()
    all_ckpts_str = all_ckpts.join(" ")
    sprobe_layers = structural_probe_layers.join(",")

    // We need to extract encodings for separate train and dev sentences.
    sentence_files = [
        params.structural_probe_train_path,
        params.structural_probe_dev_path,
    ]
    sentence_files_str = sentence_files.join(" ")

    """
#!/usr/bin/bash
for ckpt in ${all_ckpts_str}; do
    for sentence_file in ${sentence_files_str}; do
        extract_features.py \
            --input_file=\$sentence_file \
            --output_file=encodings-\$ckpt.hdf5 \
            --vocab_file=${bert_base_dir}/vocab.txt \
            --bert_config_file=${bert_base_dir}/bert_config.json \
            --init_checkpoint=model.ckpt-\$ckpt \
            --layers="${sprobe_layers}" \
            --max_seq_length=96 \
            --batch_size=64 \
            --output_format=hdf5
    done
done
    """
}

// Expand hdf5 encodings into individual identifier + hdf5 files
encodings_sprobe.flatMap {
    els -> els[1].collect {
        fs -> [[els[0], (f.name =~ /-(\d+).jsonl/)[0][1]].join("-"), fs]
    }
}.set { encodings_sprobe_flat }

/**
 * Train and evaluate structural probe for each checkpoint.
process runStructuralProbe {
    label "medium"
    container params.structural_probes_container
    publishDir "${params.outdir}/structural-probe"

    input:
    set ckpt_id, file(encodings) into encodings_sprobe

    output:
    set ckpt_id, file("dev.uuas"), file("dev.spearman") into sprobe_results

    script:
    // TODO extract train encoding
    // TODO extract dev encodings
    // TODO prepare final YAML spec, save to temporary file
    yaml_path = null

    """
#!/usr/bin/bash
run_experiment.py --train-probe 1 ${yaml_path}
    """
}
