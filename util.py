"""
Data analysis tools shared across scripts and notebooks.
"""

import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st


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
    
    subj_perf = df.groupby("type").apply(lambda sub_df: sub_df.reset_index(level=0, drop=True).mean(axis=1)).T
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
    
    results = []
    for model1, model2 in pairs:
        w_stat, p_val = st.wilcoxon(model_preds[model1]["rank"], model_preds[model2]["rank"])
        results.append((model1, model2, w_stat, p_val))
        
    results = pd.DataFrame(results, columns=["model1", "model2", "w_stat", "p_val"]) \
        .set_index(["model1", "model2"])
    
    if correct_bonferroni:
        correction = len(pairs)
        results["p_val_corrected"] = results.p_val / correction
        
    return results