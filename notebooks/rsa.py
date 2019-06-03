import itertools

from scipy import stats as st
from scipy.spatial.distance import pdist
from tqdm import tqdm

import pandas as pd


def rsa_encodings(encodings_dict, pairs=None, collapse_fn=None):
    """
    Compute representational similarity metrics on the given encodings.
    
    Arguments:
        pairs: encoding pairs (keys of `encodings_dict`) to compare. If `None`, all possible pairs are evaluated.
        collapse_fn: if not `None`, store the results of each pairwise analysis not under the key `(model1, model2)` (where `model1`, `model2` are keys of `pairs`), but rather `(collapse_fn(model1), collapse_fn(model2))`.
    """
    
    if pairs is None:
        pairs = list(itertools.combinations(encodings_dict.keys(), 2))
    
    # Cache distance matrices.
    dist_matrices = {}
    
    rsa_sims = []
    for m1_key, m2_key in tqdm(pairs):
        dists1 = dist_matrices.get(m1_key)
        if dists1 is None:
            dists1 = pdist(encodings_dict[m1_key])
            dist_matrices[m1_key] = dists1

        dists2 = dist_matrices.get(m2_key)
        if dists2 is None:
            dists2 = pdist(encodings_dict[m2_key])
            dist_matrices[m2_key] = dists2

        pearson_coef, _ = st.spearmanr(dists1, dists2)

        if collapse_fn is not None:
            m1_key = collapse_fn(m1_key)
            m2_key = collapse_fn(m2_key)
            
        rsa_sims.append((m1_key, m2_key, pearson_coef))
        
    rsa_sims = pd.DataFrame(rsa_sims, columns=["model1", "model2", "pearsonr"])
    return rsa_sims