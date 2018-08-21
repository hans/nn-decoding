"""
Render a heat-map describing the relationship between different encodings.
"""

from argparse import ArgumentParser
import itertools
import logging
import multiprocessing
import os.path
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import seaborn as sns
from tqdm import tqdm, trange


def eval_encodings_cca(enc1, enc2):
  cv = KFold(n_splits=4)
  corr_results = []

  for train_idxs, test_idxs in tqdm(cv.split(enc1), total=cv.get_n_splits(enc1),
                                    desc="CV splits"):
    from rcca import CCA
    # TODO sanity check regularization constant s.t. CCA on self yields reasonable numbers
    cca = CCA(kernelcca=False, reg=1e-6, numCC=128, verbose=False)
    cca.train([enc1[train_idxs], enc2[train_idxs]])

    print(np.mean(cca.validate([enc1[train_idxs], enc2[train_idxs]])[0]))
    enc1_pred_corrs, enc2_pred_corrs = cca.validate([enc1[test_idxs], enc2[test_idxs]])
    # TODO projection weighting
    corr_results.append(np.mean(enc2_pred_corrs))

  return np.mean(corr_results)


def eval_encodings_rdm(encodings, enc1_key, enc2_key,
                       n_bootstrap_samples=100, sentences=None):
  """
  Evaluate the similarity between two encodings `e1, e2` as follows:

  1. Align the paired representations of `e1` and `e2` to have maximal
     similarity (maximal dot product) via regularized CCA. (This CCA is
     cross-validated to prevent overfitting.)
  2. Estimate a Spearman correlation coefficient relating the pairwise
     similarity judgments predicted by the two encodings by a bootstrap. (In
     other words, bootstrap-estimate a representational similarity analysis; in
     other words; bootstrap-estimate the difference between the
     representational dissimilarity matrices (RDMs) of the two aligned
     encodings.) representations.

  Args:
    encodings: Dictionary mapping from encoding name to n_examples * d matrices
    enc1_key: string key into `encodings`
    enc2_key: string key into `encodings`
    n_bootstrap_samples: Number of samples to take in estimating bootstrap.
    sentences: Optional `n_examples` array of sentences, for debugging

  Returns:
    spearman_coefs: Bootstrap estimates of the Spearman coefficient relating
      the pairwise similarity rankings predicted by CCA-aligned forms of `enc1`
      and `enc2`
  """
  enc1 = encodings[enc1_key]
  enc2 = encodings[enc2_key]
  assert enc1.shape[0] == enc2.shape[0]

  # First align with CCA.
  from rcca import CCA, CCACrossValidate
  cca = CCACrossValidate(kernelcca=False, regs=[1e-2,1e-1,1.0,10.0,20.0], numCCs=[32,64,128,256])
  cca.train([enc1, enc2])
  print("Best reg: %f; best CC: %i" % (cca.best_reg, cca.best_numCC))

  enc1_aligned, enc2_aligned = cca.comps
  # Calculate pairwise distances.
  dists_X = pdist(enc1_aligned, "correlation")
  dists_Y = pdist(enc2_aligned, "correlation")

  dists_X_square = squareform(dists_X)
  dists_Y_square = squareform(dists_Y)

  if sentences is not None:
    # DEBUG: List some of the most similar inputs
    sent_combinations = list(itertools.combinations(range(len(enc1)), 2))
    high_sim_X = np.argsort(dists_X)
    high_sim_Y = np.argsort(dists_Y)

    out_path = "sim_%s_%s.csv" % (enc1_key, enc2_key)
    with open(out_path, "w") as out_f:
      for i, high_sim_X_idx in enumerate(high_sim_X):
        sent1, sent2 = sent_combinations[high_sim_X_idx]
        out_f.write("%s,%d,%f,\"%s\",\"%s\"\n" % (enc1_key, i, dists_X_square[sent1, sent2],
                                                  sentences[sent1], sentences[sent2]))
      for i, high_sim_Y_idx in enumerate(high_sim_Y):
        sent1, sent2 = sent_combinations[high_sim_Y_idx]
        out_f.write("%s,%d,%f,\"%s\",\"%s\"\n" % (enc2_key, i, dists_Y_square[sent1, sent2],
                                                  sentences[sent1], sentences[sent2]))

  # # Bootstrap estimate the Spearman coefficient.
  # spearman_coefs = []
  # for _ in trange(n_bootstrap_samples):
  #   idxs = np.random.choice(len(enc1), size=len(enc1), replace=True)
  #   dists_X_sample = dists_X_square[np.ix_(idxs, idxs)]
  #   dists_Y_sample = dists_Y_square[np.ix_(idxs, idxs)]

  #   # Compute Spearman coefficient on condensed / non-redundant form.
  #   sample_coef, _ = spearmanr(squareform(dists_X_sample), squareform(dists_Y_sample))
  #   spearman_coefs.append(sample_coef)

  spearman_coef, _ = spearmanr(dists_X, dists_Y)
  print("\t", enc1_key, enc2_key, spearman_coef)
  return [spearman_coef]


def eval_pair(inputs):
  enc1, enc2, encodings, sentences = inputs
  # Multiprocessing task function.
  if enc1 == enc2:
    return enc1, enc2, (1.0, 1.0)
  elif enc1 < enc2:
    # Measure should be symmetric -- only fill lower triangle of heatmap matrix.
    return enc1, enc2, (None, None)
  else:
    coefs = eval_encodings_rdm(encodings, enc1, enc2, sentences=sentences)
    # Calculate 95% CI bounds
    lower_bound, upper_bound = np.percentile(coefs, (0.5, 0.95))
    return enc1, enc2, (lower_bound, upper_bound)


def main(args):
  encodings, encoding_keys = {}, {}
  for encoding_path in args.encodings:
    encodings_i = np.load(encoding_path)
    encoding_key = os.path.basename(encoding_path)
    encoding_key = encoding_key[:encoding_key.rindex(".")]

    if args.encoding_project is not None and args.encoding_project < encodings_i.shape[1]:
      logger.info("Projecting %s to dimension %i with PCA", encoding_path, args.encoding_project)
      pca = PCA(args.encoding_project).fit(encodings_i)
      logger.info("PCA explained variance: %f", sum(pca.explained_variance_ratio_) * 100)
      encodings_i = pca.transform(encodings_i)

    encodings[encoding_key] = encodings_i
    encoding_keys[encoding_path] = encoding_key

  sentences = None
  if args.sentences_path is not None:
    with open(args.sentences_path, "r") as sentences_f:
      sentences = [line.strip() for line in sentences_f]

  # Prepare output structures
  assert len(set(enc.shape[0] for enc in encodings.values())) == 1
  # Make sure to maintain ordering of the encodings given in the CLI arguments.
  encoding_index = {encoding_keys[enc_path]: i for i, enc_path in enumerate(args.encodings)}
  heatmap_mat_lower_bound = np.empty((len(encodings), len(encodings)))
  heatmap_mat_upper_bound = np.empty_like(heatmap_mat_lower_bound)

  # Prepare multiprocessing jobs
  pool = multiprocessing.Pool(processes=args.num_processes)
  job_inputs = [(enc1, enc2, encodings, sentences) for enc1, enc2
                in itertools.product(sorted(encodings.keys()), repeat=2)]
  jobs = pool.imap_unordered(eval_pair, job_inputs)

  # Join jobs and update matrices
  with tqdm(total=len(job_inputs)) as pbar:
    for enc1, enc2, val in tqdm(jobs):
      pbar.update()
      lower_bound, upper_bound = val
      if lower_bound is None:
        continue

      heatmap_mat_lower_bound[encoding_index[enc1], encoding_index[enc2]] = lower_bound
      heatmap_mat_upper_bound[encoding_index[enc1], encoding_index[enc2]] = upper_bound

  if args.names is not None:
    names = args.names.strip().split(",")
    assert len(names) == len(args.encodings)
  else:
    names = list(map(str, range(1, len(args.encodings) + 1)))

  # Calculate heatmap statistics / render figures.
  for heatmap_mat, heatmap_name in zip([heatmap_mat_lower_bound, heatmap_mat_upper_bound],
                                       ["lower_bound", "upper_bound"]):
    # Copy lower triangle of matrix to upper triangle.
    heatmap_mat.T[np.tril_indices(len(heatmap_mat), -1)] = \
        heatmap_mat[np.tril_indices(len(heatmap_mat), -1)]

    df = pd.DataFrame(heatmap_mat, index=names, columns=names)
    df.mean(axis=1).to_csv("averages_%s.csv" % heatmap_name)
    df.to_csv("heatmap_%s.csv" % heatmap_name)

    # Only plot lower triangle.
    mask = np.zeros_like(heatmap_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(data=df, annot=True, square=True, mask=mask)
    plt.xticks(weight="bold")
    plt.yticks(rotation=0, weight="bold")
    plt.tight_layout()
    fig.savefig("heatmap_%s.svg" % heatmap_name)


if __name__ == '__main__':
  p = ArgumentParser()
  p.add_argument("encodings", nargs="+")
  p.add_argument("--encoding_project", type=int)
  p.add_argument("--names")
  p.add_argument("--sentences_path")
  p.add_argument("-p", "--num_processes", default=1, type=int)
  main(p.parse_args())
