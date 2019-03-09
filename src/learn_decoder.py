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
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import scipy.io
from scipy.spatial import distance
from tqdm import tqdm

import util

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)

# Candidate ridge regression regularization parameters.
ALPHAS = [1, 10, .01, 100, .001, 1000, .0001, 10000, .00001, 100000, .000001, 1000000, 10000000]


def learn_decoder(images, encodings, cv=None, n_jobs=1):
  """
  Learn a decoder mapping from sentence encodings to subject brain images.
  """
  gs = GridSearchCV(Ridge(fit_intercept=False, normalize=False),
                    {"alpha": ALPHAS}, cv=cv, n_jobs=n_jobs, verbose=10)
  gs.fit(images, encodings)
  decoder = gs.best_estimator_

  L.debug("Best alpha: %f", decoder.alpha_)
  return decoder


def eval_ranks(decoder, X_test, Y_test_idxs, encodings_normed):
  """
  Evaluate a trained decoder, predicting the concepts associated with test
  imaging data.

  Returns:
    ranks: len(Y_test_idxs) * len(Y) integer matrix. Each row specifies a
      ranking over sentences computed using the decoding model, given the
      brain image corresponding to each row of Y_test_idxs.
    rank_of_correct: len(Y_test_idxs) array indicating the rank of the target
      concept for each test input.
  """
  N_test = len(X_test)
  assert N_test == len(Y_test_idxs)

  Y_pred = decoder.predict(X_test)

  # For each Y_pred, evaluate rank of corresponding Y_test example among the
  # entire collection of Ys (not just Y_test), where rank is established by
  # cosine distance.
  Y_pred /= np.linalg.norm(Y_pred, axis=1, keepdims=True)
  # n_Y_test * n_sentences
  similarities = np.dot(Y_pred, encodings_normed.T)

  # Calculate distance ranks across rows.
  orders = (-similarities).argsort(axis=1)
  ranks = orders.argsort(axis=1)
  # Find the rank of the desired vectors.
  ranks_test = ranks[np.arange(len(Y_test_idxs)), Y_test_idxs]

  return ranks, rank_test


def main(args):
  print(args)

  sentences = util.load_sentences(args.sentences_path)
  encodings = util.load_encodings(args.encoding_paths, project=args.encoding_project)
  encodings_normed = encodings / np.linalg.norm(encodings, axis=1, keepdims=True)

  assert len(encodings) == len(sentences)

  ######### Prepare to process subject.

  # Load subject data.
  subject = args.subject_name or args.brain_path.name
  subject_data = scipy.io.loadmat(str(args.brain_path / args.mat_name))
  L.info("Loaded subject %s data.", subject)

  subject_images = subject_data["examples"]
  assert len(subject_images) == len(sentences)

  ######### Prepare learning setup.

  # Track within-subject performance.
  index = pd.MultiIndex.from_product(([subject], np.arange(args.n_folds)))
  metrics = pd.DataFrame(columns=["avg_rank"], index=index)

  # Prepare nested CV.
  # Inner CV is responsible for hyperparameter optimization;
  # outer CV is responsible for prediction.
  state = int(time.time())
  inner_cv = KFold(n_splits=args.n_folds, shuffle=True, random_state=state)
  outer_cv = KFold(n_splits=args.n_folds, shuffle=True, random_state=state)

  # Prepare scoring function for outer CV.
  def scoring_fn(decoder, X_test, Y_test_idxs):
    """
    Evaluate a learned decoder on test data mapping brain images X to model
    encodings Y.

    Returns:
      avg_rank: Average distance rank of ground-truth sentence from predicted
        sentence vectors across the test data.
    """
    ranks, ranks_test = eval_ranks(decoder, X_test, Y_test_idxs, encodings_normed)
    return ranks_test.mean()

  ######## Run learning.

  X = subject_images
  Y = encodings
  Y_idxs = np.arange(len(encodings))

  # Run inner CV.
  decoder = learn_decoder(X, Y, cv=inner_cv, n_jobs=args.n_jobs)
  # Run outer CV.
  decoder_scores = cross_val_score(decoder, X, Y_idxs, cv=outer_cv)


  ######### Save results.

  metrics.loc[subject_name, np.arange(len(decoder_scores))]["avg_rank"] = decoder_scores
  metrics.to_csv(args.out_prefix + ".csv")

  # # Save per-sentence outputs.
  # predicted_ranks = list(itertools.chain.from_iterable(
  #   [(subject, idx, rank) for idx, rank in subject_ranks]
  #   for subject, subject_ranks in predicted_ranks.items()))
  # predicted_ranks = pd.DataFrame(predicted_ranks, columns=["subject", "idx", "rank"]) \
  #     .set_index(["subject", "idx"])
  # predicted_ranks.to_csv(args.out_prefix + ".pred.csv")


if __name__ == '__main__':
  p = ArgumentParser()

  p.add_argument("sentences_path", type=Path)
  p.add_argument("brain_path", type=Path)
  p.add_argument("encoding_paths", type=Path, nargs="+")
  p.add_argument("--encoding_project", type=int)
  p.add_argument("--n_folds", type=int, default=12)
  p.add_argument("--mat_name", default="examples_384sentences.mat")
  p.add_argument("--out_prefix", default="decoder_perf")
  p.add_argument("--subject_name", help="By default, basename of brain_path")
  p.add_argument("--n_jobs", type=int, default=1)

  main(p.parse_args())
