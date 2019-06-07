"""
Evaluate the predictions of super-voxel encoder models.
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import util


ROOT = Path(__file__).absolute().parent.parent
BRAIN_DATA = ROOT / "data" / "brains"
MAT_FILE = "examples_384sentences.mat"

# string template for decoder prediction files
DECODER_PRED_TEMPLATE = str(ROOT / "models" / "encoders" / "encodings.finetune-250.uncased_L-12_H-768_A-12.%s-run%i-250-layer%i-%s.whole-brain-down3.pred.npy")



def get_errors(model, subject, subject_images, runs, layers):
  """
  Get prediction errors for the given model across runs and at different layers.
  """
  predictions = [[np.load(DECODER_PRED_TEMPLATE % (model, run, layer, subject))
                  for layer in layers]
                 for run in runs]
  errors = np.array([[mean_squared_error(subject_images, predictions_l, multioutput="raw_values")
                      for predictions_l in run_predictions]
                     for run_predictions in predictions])

  # Average super-voxelwise errors across model runs.
  errors = errors.mean(axis=0)

  return errors
  
  
def threshold_and_normalize_errors(errors, coords, n=1000, keep_layers=None):
  """
  Args:
      errors: `n_layers * n_voxels` error matrix
      coords: `n_voxels * 3` integer matrix, describing coordinates of each super-voxel in subject brain

  Returns:
      errors_filtered: `n_layers * n` error matrix, normalized per super-voxel
      coords_filtered: `n * 3` matrix specifying coordinates of each retained super-voxel
  """
  if keep_layers:
    errors = errors[keep_layers, :]
  best = errors.min(axis=0).argsort()[:n]
  errors_filtered, coords_filtered = errors[:, best], coords[best]

  # Normalize errors across super-voxels
  errors_filtered -= errors_filtered.min()
  errors_filtered /= errors_filtered.max()

  # Normalize errors within super-voxel
  if errors_filtered.shape[0] > 1:
    errors_filtered -= errors_filtered.min(axis=0)
    errors_filtered /= errors_filtered.max(axis=0)

  return errors_filtered, coords_filtered


def evaluate_subject(subject, models, runs, layers,
                     downsample=3, threshold=1000):
  """
  Evaluate encoders learned for the given subject.
  """
  # Load subject brain data.
  brain_path = BRAIN_DATA / subject / MAT_FILE
  subject_images, coords = util.load_brain_data(
    brain_path, downsample=downsample, ret_coords=True)
  # Normalize as in encoder.
  subject_images -= subject_images.mean(axis=0)
  subject_images /= np.linalg.norm(subject_images, axis=1, keepdims=True)
  
  # Evaluate error.
  all_errors = {model: get_errors(model, subject, subject_images, runs, layers)
                for model in tqdm(models, desc="Models")}
  
  # Get thresholded errors and coordinates.
  thresholded_errors = {model: threshold_and_normalize_errors(errors, coords, n=threshold)
                        for model, errors in all_errors.items()}
  
  return all_errors, thresholded_errors


def main(subjects, models, runs, layers, args):
  all_errors, thresholded_errors = {}, {}
  for subject in tqdm(subjects, desc="Subjects"):
    all_errors[subject], thresholded_errors[subject] = \
      evaluate_subject(subject, models, runs, layers,
                       downsample=args.downsample, threshold=args.threshold)
    
  with args.out_file.open("wb") as out_f:
    pickle.dump({"all_errors": all_errors,
                 "thresholded_errors": thresholded_errors},
                out_f)
  
  
  
if __name__ == "__main__":
  p = ArgumentParser()
  p.add_argument("out_file", type=Path)
  p.add_argument("--downsample", type=int, default=3)
  p.add_argument("--threshold", help="Number of super-voxels to include in final analyses",
                 type=int, default=100)
  p.add_argument("--subjects", type=lambda x: x.split(","),
                 default=[subject_dir.name for subject_dir in BRAIN_DATA.iterdir()
                          if subject_dir.is_dir()])
  p.add_argument("--models", type=lambda x: x.split(","),
                 default=["MNLI", "SST", "QQP", "SQuAD", "LM_scrambled", "LM_pos"])
  p.add_argument("--layers", type=lambda x: list(map(int, x.split(","))),
                 default=[2, 5, 8, 9, 10, 11])
  p.add_argument("--runs", type=lambda x: list(map(int, x.split(","))),
                 default=[1, 2])
  
  args = p.parse_args()
  
  main(args.subjects, args.models, args.runs, args.layers, args)