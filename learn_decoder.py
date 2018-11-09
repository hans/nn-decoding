#!/usr/bin/env python
"""
Learn a decoder mapping from functional imaging data to target model
representations.
"""
from argparse import ArgumentParser
from collections import defaultdict
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import scipy.io
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def learn_decoder(images, encodings):
  """
  Learn a decoder mapping from sentence encodings to subject brain images.
  """
  ridge = RidgeCV(
      alphas=[1, 10, .01, 100, .001, 1000, .0001, 10000, .00001, 100000, .000001, 1000000, 10000000],
      fit_intercept=False
  )
  ridge.fit(images, encodings)
  logger.debug("Best alpha: %f", ridge.alpha_)
  return ridge


def iter_folds(subject_data, encodings, n_folds=18):
  """
  Yield CV folds of a dataset.
  """
  N = len(subject_data)
  assert N == len(encodings)

  fold_size = N // n_folds
  for fold_i in range(n_folds):
    # NB doesn't handle oddly sized final folds
    idxs = np.arange(fold_i * fold_size, (fold_i + 1) * fold_size)
    mask = np.zeros(N, dtype=np.bool)
    mask[idxs] = True

    train_imaging = subject_data[~mask]
    train_semantic = encodings[~mask]

    test_imaging = subject_data[mask]
    test_semantic = encodings[mask]
    target_semantic_idxs = idxs

    yield (train_imaging, train_semantic), (test_imaging, test_semantic, target_semantic_idxs)


def eval_ranks(clf, test_imaging, target_semantic_idxs, encodings):
  """
  Evaluate a trained classifier, predicting the concepts associated with test imaging data.

  Returns:
    rankings: N_test * N_concepts integer matrix. Each row is a list of concept IDs
      ranked using the model.
    rank_of_correct: N_test vector indicating the rank of the target concept for each
      test input.
  """
  N_test = len(test_imaging)
  assert N_test == len(target_semantic_idxs)

  # yields an N_test * D_sem matrix
  pred_semantic = clf.predict(test_imaging)

  # calculate pairwise similarities with semantic data
  # yields an N_test * C matrix
  pred_semantic /= np.linalg.norm(pred_semantic, axis=1, keepdims=True)
  encodings_normed = encodings / np.linalg.norm(encodings, axis=1, keepdims=True)
  similarities = np.dot(pred_semantic, encodings_normed.T)

  # Rank similarities row-wise in descending order
  rankings = np.argsort(-similarities, axis=1)
  # Find the rank of the desired concept vector
  matches = np.equal(rankings, target_semantic_idxs[:, np.newaxis])
  rank_of_correct = np.argmax(matches, axis=1)
  return rankings, rank_of_correct


def run_fold(fold, encodings, permute_targets=False):
  """
  Train and evaluate a classifier on the given CV setup.

  Args:
    fold:
    encodings: N_concepts * encoding_dim matrix
    permute_targets: If `True`, permute the target semantic idxs in order to
      evaluate baseline performance.

  Returns:
    test_idxs: N_test vector indicating the index of each target
      concept in the whole list of concepts
    rankings: N_test * N_concepts integer matrix. Each row is a list of concept
      IDs ranked using the model.
    rank_of_correct: N_test vector indicating the rank of the target concept
      for each test input.
  """
  (train_imaging, train_semantic), (test_imaging, test_semantic, target_semantic_idxs) = fold
  clf = learn_decoder(train_imaging, train_semantic)

  if permute_targets:
    target_semantic_idxs = target_semantic_idxs.copy()
    np.random.shuffle(target_semantic_idxs)

  rankings, rank_of_correct = eval_ranks(clf, test_imaging, target_semantic_idxs, encodings)
  return target_semantic_idxs, rankings, rank_of_correct


def main(args):
  print(args)

  with open(args.sentences_path, "r") as sentences_f:
    sentences = [line.strip() for line in sentences_f]

  encodings = np.load(args.encoding_path)
  logger.info("Loaded encodings of size %s.", encodings.shape)

  if args.encoding_project is not None:
    logger.info("Projecting encodings to dimension %i with PCA", args.encoding_project)

    if encodings.shape[1] < args.encoding_project:
      logger.warn("Encodings are already below requested dimensionality: %i < %i"
                  % (encodings.shape[1], args.encoding_project))
    else:
      pca = PCA(args.encoding_project).fit(encodings)
      logger.info("PCA explained variance: %f", sum(pca.explained_variance_ratio_) * 100)
      encodings = pca.transform(encodings)

  assert len(encodings) == len(sentences)

  ######### Prepare to process subjects.
  subject_paths = [item for item in Path(args.subject_dir).glob("*") if item.is_dir()]

  # Track within-subject performance on test folds and permuted test folds.
  index_vals = itertools.product(["ridge", "ridge_permute"], [p.name for p in subject_paths])
  mar_metrics = pd.DataFrame(columns=['mar_fold_%i' % i for i in range(args.n_folds)],
                             index=pd.MultiIndex.from_tuples(index_vals, names=("type", "subject")))
  predicted_ranks = defaultdict(list)

  for path in tqdm(subject_paths, desc="subjects"):
    # Load subject data.
    subject = path.name
    subject_data = scipy.io.loadmat(str(path / args.mat_name))
    logger.info("Loaded subject %s data.", subject)

    subject_images = subject_data["examples"]
    assert len(subject_images) == len(sentences)

    # Track within-subject performance on test fold and permuted test fold.
    perf_test, perf_permute = [], []

    folds = iter_folds(subject_images, encodings, n_folds=args.n_folds)
    for i, fold in enumerate(tqdm(folds, total=args.n_folds, desc="%s folds" % subject)):
      # Calculate predicted ranks and MAR for this subject on this fold
      test_idxs, _, rank_of_correct = run_fold(fold, encodings)
      predicted_ranks[subject].extend(list(zip(test_idxs, rank_of_correct)))
      perf_test.append(rank_of_correct.mean())
      tqdm.write("Fold %2i:\t\tmin %3.1f\tmean %3.1f\tmed %3.1f\tmax %3.1f" %
                 (i, rank_of_correct.min(), rank_of_correct.mean(),
                  np.median(rank_of_correct), rank_of_correct.max()))

      # Baseline: test-time prediction with permuted labels
      _, _, rank_of_correct_permute = run_fold(fold, encodings, permute_targets=True)
      perf_permute.append(rank_of_correct_permute.mean())
      tqdm.write("Fold permuted %2i:\tmin %3.1f\tmean %3.1f\tmed %3.1f\tmax %3.1f" %
                 (i, rank_of_correct_permute.min(), rank_of_correct_permute.mean(),
                  np.median(rank_of_correct_permute), rank_of_correct_permute.max()))

    # Save MAR results.
    mar_metrics.loc["ridge", subject] = perf_test
    mar_metrics.loc["ridge_permute", subject] = perf_permute

  # Save mean-average-rank metrics.
  print(mar_metrics)
  mar_metrics.to_csv(args.out_prefix + ".csv")

  # Save per-sentence outputs.
  predicted_ranks = list(itertools.chain.from_iterable(
    [(subject, idx, rank) for idx, rank in subject_ranks]
    for subject, subject_ranks in predicted_ranks.items()))
  predicted_ranks = pd.DataFrame(predicted_ranks, columns=["subject", "idx", "rank"]) \
      .set_index(["subject", "idx"])
  predicted_ranks.to_csv(args.out_prefix + ".pred.csv")


if __name__ == '__main__':
  p = ArgumentParser()

  p.add_argument("sentences_path")
  p.add_argument("encoding_path")
  p.add_argument("subject_dir")
  p.add_argument("--encoding_project", type=int)
  p.add_argument("--n_folds", type=int, default=18)
  p.add_argument("--mat_name", default="examples_384sentences.mat")
  p.add_argument("--out_prefix", default="decoder_perf")

  main(p.parse_args())
