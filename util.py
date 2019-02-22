"""
Data analysis tools shared across scripts and notebooks.
"""

from collections import defaultdict
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st


def load_sentences(sentence_path="data/stimuli_384sentences.txt"):
    with open(sentence_path, "r") as f:
        sentences = [line.strip() for line in f]
    return sentences


def load_decoding_perf(name, results_path, ax=None):
    """
    Load and render a DataFrame describing decoding performance for a particular representation.
    
    Args:
        name: Name (for rendering)
        results_path: path to CSV decoding results
    """
    df = pd.read_csv(results_path, index_col=[0, 1])
    
    if ax is not None:
        sns.violinplot(x="subject", y="value", hue="type",
                       data=df.reset_index().melt(id_vars=["subject", "type"],
                                                  value_vars=["mar_fold_%i" % i for i in range(18)]), 
                       ax=ax)
        ax.set_ylim((40, 260))
        ax.set_ylabel("average rank")
        ax.set_title("%s: Within-subject MAR" % name)
    
    subj_perf = df.groupby("type").apply(
        lambda sub_df: sub_df.reset_index(level=0, drop=True).T.agg(["mean", "sem"]))
    #subj_perf.plot.bar(title="%s: Within-subject MAR" % name)
    
    return subj_perf


def wilcoxon_rank_preds(models, correct_bonferroni=True, pairs=None):
    """
    Run Wilcoxon rank tests comparing the ranks of correct sentence representations in predictions 
    from two or more models.
    """
    if pairs is None:
        pairs = list(itertools.combinations(models.keys(), 2))
        
    model_preds = {model: pd.read_csv("perf.384sentences.%s.pred.csv" % path).sort_index()
                   for model, path in models.items()}
    
    subjects = next(iter(model_preds.values())).subject.unique()
    
    results = []
    for model1, model2 in pairs:
        w_stat, p_val = st.wilcoxon(model_preds[model1]["rank"], model_preds[model2]["rank"])
        results.append((model1, model2, w_stat, p_val))
        
    results = pd.DataFrame(results, columns=["model1", "model2", "w_stat", "p_val"]) \
        .set_index(["model1", "model2"])
    
    if correct_bonferroni:
        correction = len(pairs) * len(subjects)
        print(0.01 / correction, len(subjects))
        results["p_val_corrected"] = results.p_val / correction
        
    return results


def load_bert_finetune_metadata(savedir, checkpoint_steps=None):
    """
    Load metadata for an instance of a finetuned BERT model.
    """
    savedir = Path(savedir)
    
    import tensorflow as tf
    from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
    try:
        ckpt = NewCheckpointReader(str(savedir / "model.ckpt"))
    except tf.errors.NotFoundError:
        if checkpoint_steps is None:
            raise
        ckpt = NewCheckpointReader(str(savedir / ("model.ckpt-%i" % checkpoint_steps[-1])))
    
    ret = {}
    try:
        ret["global_steps"] = ckpt.get_tensor("global_step")
        ret["output_dims"] = ckpt.get_tensor("output_bias").shape[0]
    except tf.errors.NotFoundError:
        ret.setdefault("global_steps", np.nan)
        ret.setdefault("output_dims", np.nan)
    
    ret["steps"] = defaultdict(dict)
    
    # Load training events data.
    try:
        events_file = next(savedir.glob("events.*"))
    except StopIteration:
        # no events data -- skip
        pass
    else:
        total_global_norm = 0.
        first_loss, cur_loss = None, None
        tags = set()
        for e in tf.train.summary_iterator(str(events_file)):
            for v in e.summary.value:
                tags.add(v.tag)
                if v.tag == "grads/global_norm":
                    total_global_norm += v.simple_value
                elif v.tag in ["loss_1", "loss"]:
                    # SQuAD output stores loss in `loss` key;
                    # classifier stores in `loss_1` key.
                    
                    if e.step == 1:
                        first_loss = v.simple_value
                    cur_loss = v.simple_value
                    
            if checkpoint_steps is None or e.step in checkpoint_steps:
                ret["steps"][e.step].update({
                    "total_global_norms": total_global_norm,
                    "train_loss": cur_loss,
                    "train_loss_norm": cur_loss / ret["output_dims"]
                })
                
        ret["first_train_loss"] = first_loss
        ret["first_train_loss_norm"] = first_loss / ret["output_dims"]
        
    # Load eval events data.
    try:
        eval_events_file = next(savedir.glob("eval/events.*"))
    except StopIteration:
        # no eval events data -- skip
        pass
    else:
        tags = set()
        eval_loss, eval_accuracy = None, None
        for e in tf.train.summary_iterator(str(eval_events_file)):
            for v in e.summary.value:
                tags.add(v.tag)
                if v.tag == "eval_loss":
                    eval_loss = v.simple_value
                elif v.tag == "eval_accuracy":
                    eval_accuracy = v.simple_value
                
            if checkpoint_steps is None or e.step in checkpoint_steps:
                ret["steps"][e.step].update({
                    "eval_accuracy": eval_accuracy,
                    "eval_loss": eval_loss,
                })
                
    return ret