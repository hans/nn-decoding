"""
Data analysis tools shared across scripts and notebooks.
"""

from collections import defaultdict
import itertools
import logging
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg", warn=False)
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ModuleNotFoundError:
    pass
import scipy.io as io
import scipy.stats as st
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from tqdm import tqdm

import gordon

L = logging.getLogger(__name__)


def load_sentences(sentence_path="data/sentences/stimuli_384sentences.txt"):
    with open(sentence_path, "r") as f:
        sentences = [line.strip() for line in f]
    return sentences


def load_encodings(paths, project=None):
  encodings = []
  for encoding_path in paths:
    encodings_i = np.load(encoding_path)
    L.info("%s: Loaded encodings of size %s.", encoding_path, encodings_i.shape)

    if project is not None:
      L.info("Projecting encodings to dimension %i with PCA", project)

      if encodings_i.shape[1] < project:
        L.warn("Encodings are already below requested dimensionality: %i < %i"
                    % (encodings_i.shape[1], project))
      else:
        pca = PCA(project).fit(encodings_i)
        L.info("PCA explained variance: %f", sum(pca.explained_variance_ratio_) * 100)
        encodings_i = pca.transform(encodings_i)

    encodings.append(encodings_i)

  encodings = np.concatenate(encodings, axis=1)
  return encodings

def load_full_brain_data(path):
  subject_data = loadmat(path)
  return subject_data

def project_roi_images(roi_images):
    L.info("Original shape of images: {}".format(roi_images.shape))
    dim = min(roi_images.shape)
    pca = PCA(dim).fit(roi_images)
    while sum(pca.explained_variance_ratio_) > 0.95:
        dim = int(dim / 1.5)
        pca = PCA(dim).fit(roi_images)
    L.info("Projected to %d dimensions", dim)
    L.info("PCA explained variance: %f", sum(pca.explained_variance_ratio_) * 100)
    roi_images = pca.transform(roi_images)
    return roi_images


def downsample_subject_images(subject_data, block_shape=(4, 4, 4)):
  """
  Downsample the brain images for a given subject by averaging within local
  regions, and retain only those regions for which all contained voxels are
  members of any ROI.

  Returns:
    examples: `N_examples * d_downsampled` ndarray brain images
  """
  # Downsample by taking means across local blocks.
  downsample_fn = np.mean

  # First prepare ROI mask.
  # Convert ROI mask to a 3D volume.
  roi_vol = gordon.reconstruct_3D_roi(subject_data)
  # Drop ROI information -- we just need to know which voxels are part of the
  # whole-brain image and which are not. We effectively get a head mask from
  # this.
  roi_vol = roi_vol != 0
  # Now downsample ROI volume. Each resulting cell tells us, for each
  # corresponding region, how many of the contained cells are within the head
  # mask.
  roi_vol = block_reduce(roi_vol, block_shape, func=np.mean)
  # Grab indices of reduced matrix that we want to retain
  save_indices = (roi_vol > 0.6).nonzero()

  # Now downsample each brain image and save only indices of interest.
  new_examples = np.array(
      [block_reduce(volume, block_shape, downsample_fn)[save_indices]
       for volume in gordon.reconstruct_3D_examples(subject_data)])
  return new_examples


def load_brain_data(path, project=None, downsample=None):
  subject_data = loadmat(path)
  subject_images = subject_data["examples"]

  if downsample is not None and project is not None:
    L.warn("Downsampling and down-projecting afterwards. Does this make sense?")

  if downsample is not None:
    L.info("Downsampling brain images with %i-voxel cubes", downsample)
    old_shape = subject_images.shape
    subject_images = downsample_subject_images(subject_data, block_shape=(downsample,) * 3)
    L.info("Downsampled shape: %s (old shape %s)", subject_images.shape, old_shape)

  if project is not None:
    L.info("Projecting brain images to dimension %i with PCA", project)
    if subject_images.shape[1] < project:
      L.warn("Images are already below requested dimensionality: %i < %i"
             % (subject_images.shape[1], project))
    else:
      pca = PCA(project).fit(subject_images)
      L.info("PCA explained variance: %f", sum(pca.explained_variance_ratio_) * 100)
      subject_images = pca.transform(subject_images)

  return subject_images


def load_decoding_perfs(results_dir, glob_prefix=None):
    """
    Load and render a DataFrame describing decoding performance across models,
    model runs, and subjects.

    Args:
        results_dir: path to directory containing CSV decoder results
    """
    decoder_re = re.compile(r"\.(\w+)-run(\d+)-(\d+)-([\w\d]+)\.csv$")

    results = {}
    result_keys = ["model", "run", "step", "subject"]
    for csv in tqdm(list(Path(results_dir).glob("%s*.csv" % (glob_prefix or ""))),
                    desc="Loading perf files"):
      model, run, step, subject = decoder_re.findall(csv.name)[0]
      try:
        df = pd.read_csv(csv, usecols=["mse", "r2",
                                       "rank_median", "rank_mean",
                                       "rank_min", "rank_max"])
      except ValueError:
        continue

      results[model, int(run), int(step), subject] = df

    if len(results) == 0:
        raise ValueError("No valid csv outputs found.")

    ret = pd.concat(results, names=result_keys)
    # drop irrelevant CSV row ID level
    ret.index = ret.index.droplevel(-1)
    return ret


def load_decoding_preds(results_dir, glob_prefix=None):
    """
    Load decoder predictions into a dictionary organized by decoder properties:
    decoder target model, target model run, target model run training step,
    and source subject image.
    """
    decoder_re = re.compile(r"\.(\w+)-run(\d+)-(\d+)-([\w\d]+)\.pred\.npy$")

    results = {}
    for npy in tqdm(list(Path(results_dir).glob("%s*.pred.npy" % (glob_prefix or ""))),
                    desc="Loading prediction files"):
        model, run, step, subject = decoder_re.findall(npy.name)[0]
        results[model, int(run), int(step), subject] = np.load(npy)

    if len(results) == 0:
        raise ValueError("No valid npy pred files found.")

    return results


def eval_ranks(Y_pred, idxs, encodings, encodings_normed=True):
  """
  Run a rank evaluation on predicted encodings `Y_pred` with dataset indices
  `idxs`.

  Args:
    Y_pred: `N_test * n_dim`-matrix of predicted encodings for some
      `N_test`-subset of sentences
    idxs: `N_test`-length array of dataset indices generating each of `Y_pred`
    encodings: `M * n_dim`-matrix of dataset encodings. The perfect decoder
      would predict `Y_pred[idxs] == encoding[idxs]`.

  Returns:
    ranks: `N_test * M` integer matrix. Each row specifies a
      ranking over sentences computed using the decoding model, given the
      brain image corresponding to each row of Y_test_idxs.
    rank_of_correct: `N_test` array indicating the rank of the target
      concept for each test input.
  """
  N_test = len(Y_pred)
  assert N_test == len(idxs)

  # TODO implicitly coupled to decoder normalization -- best to factor this
  # out!
  if encodings_normed:
    Y_pred -= Y_pred.mean(axis=0)
    Y_pred /= np.linalg.norm(Y_pred, axis=1, keepdims=True)

  # For each Y_pred, evaluate rank of corresponding Y_test example among the
  # entire collection of Ys (not just Y_test), where rank is established by
  # cosine distance.
  # n_Y_test * n_sentences
  similarities = np.dot(Y_pred, encodings.T)

  # Calculate distance ranks across rows.
  orders = (-similarities).argsort(axis=1)
  ranks = orders.argsort(axis=1)
  # Find the rank of the desired vectors.
  ranks_test = ranks[np.arange(len(idxs)), idxs]

  return ranks, ranks_test


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

    results = {}
    for model1, model2 in pairs:
        m1_preds, m2_preds = model_preds[model1], model_preds[model2]
        m_preds = m1_preds.join(m2_preds["rank"], rsuffix="_m2")
        pair_results = m_preds.groupby("subject").apply(lambda xs: st.wilcoxon(xs["rank"], xs["rank_m2"])) \
            .apply(lambda ys: pd.Series(ys, index=("w_stat", "p_val")))

        results[model1, model2] = pair_results

    results = pd.concat(results, names=["model1", "model2"]).sort_index()

    if correct_bonferroni:
        correction = len(results)
        print(0.01 / correction, len(results))
        results["p_val_corrected"] = results.p_val * correction

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
        print("Missing training events file in savedir:", savedir)
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
        print("Missing eval events data in savedir:", savedir)
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
                elif v.tag == "masked_lm_accuracy":
                    eval_accuracy = v.simple_value

            if checkpoint_steps is None or e.step in checkpoint_steps:
                ret["steps"][e.step].update({
                    "eval_accuracy": eval_accuracy,
                    "eval_loss": eval_loss,
                })

    return ret

# helper functions for importing .mat files into Python
# credit: https://stackoverflow.com/a/8832212

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], io.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        else:
            d[strg] = elem
    return d
