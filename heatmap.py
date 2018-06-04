"""
Render a heat-map describing the relationship between different encodings.
"""

from argparse import ArgumentParser
import itertools
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import seaborn as sns


def eval_encodings_pairwise(enc1, enc2):
  """
  Evaluate a regression model mapping from enc1 -> enc2.
  """
  ridge = RidgeCV(
      alphas=[1, 10, .01, 100, .001, 1000, .0001, 10000, 0.00001, 100000, 0.000001, 1000000],
      fit_intercept=False
  )
  ridge.fit(enc1, enc2)
  # TODO hold out?
  return ridge.score(enc1, enc2)


def main(args):
  encodings = {}
  for encoding_path in args.encodings:
    encodings_i = np.load(encoding_path)

    if args.encoding_project is not None and args.encoding_project < encodings_i.shape[1]:
      logger.info("Projecting %s to dimension %i with PCA", encoding_path, args.encoding_project)
      pca = PCA(args.encoding_project).fit(encodings_i)
      logger.info("PCA explained variance: %f", sum(pca.explained_variance_ratio_) * 100)
      encodings_i = pca.transform(encodings_i)

    encodings[encoding_path] = encodings_i

  assert len(set(enc.shape[0] for enc in encodings.values())) == 1

  encoding_index = {enc_path: i for i, enc_path in enumerate(args.encodings)}
  heatmap_mat = np.empty((len(encodings), len(encodings)))
  for enc1, enc2 in itertools.product(encodings.keys(), repeat=2):
    if enc1 == enc2:
      val = 1.0
    else:
      logger.info("Evaluating %s -> %s", enc1, enc2)
      val = eval_encodings_pairwise(encodings[enc1], encodings[enc2])
    heatmap_mat[encoding_index[enc1], encoding_index[enc2]] = val

  if args.names is not None:
    names = args.names.strip().split(",")
    assert len(names) == len(args.encodings)
  else:
    names = list(map(str, range(1, len(args.encodings) + 1)))
  df = pd.DataFrame(heatmap_mat, index=names, columns=names)
  df.mean(axis=1).to_csv("averages.csv")
  df.to_csv("heatmap.csv")
  fig = plt.figure(figsize=(6, 5))
  sns.heatmap(data=df, annot=True)
  plt.xticks(weight="bold")
  plt.yticks(rotation=0, weight="bold")
  plt.tight_layout()
  fig.savefig("heatmap.svg")


if __name__ == '__main__':
  p = ArgumentParser()
  p.add_argument("encodings", nargs="+")
  p.add_argument("--encoding_project", type=int)
  p.add_argument("--names")
  main(p.parse_args())
