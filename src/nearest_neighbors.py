from pathlib import Path

import numpy as np
from scipy.spatial import distance

import util


def eval_quant(encoding, metric="cosine"):
  # Compute pairwise cosine similarities.
  similarities = 1 - distance.pdist(encoding, metric=metric)

  return similarities


def main(args):
  sentences = util.load_sentences(args.sentences_path)
  encoding = np.load(encoding_path)

  if args.mode == "quant":
    eval_quant(encoding)
  elif args.mode == "qual":
    pass


if __name__ == '__main__':
  p = ArgumentParser()
  p.add_argument("sentences_path", type=Path)
  p.add_argument("encoding_path")
  p.add_argument("--mode", choices=["quant", "qual"], default="quant")
