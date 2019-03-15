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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
import scipy.io
from scipy.spatial import distance
from tqdm import tqdm

import util

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)

# Candidate ridge regression regularization parameters.
ALPHAS = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e1]


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
  L.info("Loading subject %s data.", subject)
  subject_images = util.load_brain_data(str(args.brain_path / args.mat_name),
                                        project=args.image_project)
  assert len(subject_images) == len(sentences)

  ######### Prepare learning setup.

  # Track within-subject performance.
  metrics = pd.DataFrame(columns=["mse", "r2"])

  # Prepare nested CV.
  # Inner CV is responsible for hyperparameter optimization;
  # outer CV is responsible for prediction.
  state = int(time.time())
  inner_cv = KFold(n_splits=args.n_folds, shuffle=True, random_state=state)
  outer_cv = KFold(n_splits=args.n_folds, shuffle=True, random_state=state)

  # Final data prep: normalize.
  X = subject_images - subject_images.mean(axis=0)
  X = X / np.linalg.norm(subject_images, axis=1, keepdims=True)
  Y = encodings - encodings.mean(axis=0)
  Y = Y / np.linalg.norm(encodings, axis=1, keepdims=True)

  # # Prepare scoring function for outer CV.
  # def scoring_fn(decoder, idxs, _):
  #   """
  #   Evaluate a learned decoder on test data, given indices of test data in the
  #   outer matrix.

  #   Returns:
  #     avg_rank: Average distance rank of ground-truth sentence from predicted
  #       sentence vectors across the test data.
  #   """
  #   ranks, ranks_test = eval_ranks(decoder, X[idxs], idxs, encodings_normed)
  #   return ranks_test.mean()

  ######## Run learning.

  # Run inner CV.
  gs = GridSearchCV(Ridge(fit_intercept=False, normalize=False),
                    {"alpha": ALPHAS}, cv=inner_cv, n_jobs=args.n_jobs, verbose=10)
  # Run outer CV.
  decoder_predictions = cross_val_predict(gs, X, Y, cv=outer_cv)

  ######### Evaluate.

  metrics.loc[subject, "mse"] = mean_squared_error(Y, decoder_predictions)
  metrics.loc[subject, "r2"] = r2_score(Y, decoder_predictions)

  ######### Save results.

  csv_path = "%s.csv" % args.out_prefix
  metrics.to_csv(csv_path)
  L.info("Wrote decoding results to %s" % csv_path)

  # Save per-sentence outputs.
  npy_path = "%s.pred.npy" % args.out_prefix
  np.save(npy_path, decoder_predictions)
  L.info("Wrote decoder predictions to %s" % npy_path)


if __name__ == '__main__':
  p = ArgumentParser()

  p.add_argument("sentences_path", type=Path)
  p.add_argument("brain_path", type=Path)
  p.add_argument("encoding_paths", type=Path, nargs="+")
  p.add_argument("--encoding_project", type=int)
  p.add_argument("--image_project", type=int)
  p.add_argument("--n_folds", type=int, default=12)
  p.add_argument("--mat_name", default="examples_384sentences.mat")
  p.add_argument("--out_prefix", default="decoder_perf")
  p.add_argument("--subject_name", help="By default, basename of brain_path")
  p.add_argument("--n_jobs", type=int, default=1)

  main(p.parse_args())
