#!/usr/bin/env python
"""
Select ROIs from whole-brain data.
"""
import numpy as np

def _to_gordon(subj_dict):
    examples = subj_dict['examples']
    roi_multimask_gordon = subj_dict['meta']['roiMultimaskGordon']
    indices_in_gordon = np.nonzero(roi_multimask_gordon)[0]
    examples_gordon = examples[:, indices_in_gordon]
    voxel_roi = roi_multimask_gordon[indices_in_gordon]
    rois_gordon = subj_dict['meta']['roisGordon']
    return examples_gordon, voxel_roi, rois_gordon

def _anat_dict(rois_gordon, group_by='roi'):
    anat_to_roi = {}
    for roi in rois_gordon:
        # format of roi: e.g. ROI241_VentralAttn_R_48.1_38.3_-9.2
        roi_id, region, hemi = roi.split('_')[:3]
        roi_num = int(roi_id[3:])
        anat = hemi if group_by == 'hemi' else '%s_%s' % (region, hemi)
        if anat in anat_to_roi:
            anat_to_roi[anat].append(roi_num)
        else:
            anat_to_roi[anat] = [roi_num]
    return anat_to_roi

def group_by_anat(subj_dict, group_by='roi'):
    examples_gordon, voxel_roi, rois_gordon = _to_gordon(subj_dict)
    anat_to_roi = _anat_dict(rois_gordon, group_by=group_by)
    anat_to_examples = {}
    for anat, rois in anat_to_roi.items():
        # get indices of columns that belong to rois associated with name
        voxels_from_rois = [i for i in range(len(voxel_roi)) if voxel_roi[i] in rois]
        # checked that these sum up to the number of columns in example_gordon
        examples_from_rois = examples_gordon[:, voxels_from_rois]
        anat_to_examples[anat] = examples_from_rois
    return anat_to_examples


def _reconstruct_3D_example(example, meta):
    '''
    Returns 3D volume for a single example, using the meta dictionary.
    '''
    vol = np.zeros(np.prod(meta['dimensions']))
    vol[meta['indicesIn3D']] = example
    vol = np.reshape(vol, meta['dimensions'])
    return vol

def reconstruct_3D(subj_dict):
    '''
    Returns 4D numpy array with 3D volume for each example
    by calling _reconstruct_3D_example.
    '''
    examples = subj_dict['examples']
    meta = subj_dict['meta']
    volumes = np.asarray(
        [_reconstruct_3D_example(example, meta) for example in examples]
    )
    return volumes
