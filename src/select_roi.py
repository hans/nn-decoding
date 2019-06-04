#!/usr/bin/env python
"""
Select ROIs from whole-brain data.
"""
from numpy import nonzero

def to_gordon(subj_dict):
    examples = subj_dict['examples']
    roi_multimask_gordon = subj_dict['meta']['roiMultimaskGordon']
    indices_in_gordon = nonzero(roi_multimask_gordon)[0]
    examples_gordon = examples[:, indices_in_gordon]
    voxel_roi = roi_multimask_gordon[indices_in_gordon]
    rois_gordon = subj_dict['meta']['roisGordon']
    return examples_gordon, voxel_roi, rois_gordon

def roi_dict(rois_gordon):
    anat_to_roi = {}
    for roi in rois_gordon:
        # e.g. ROI241_VentralAttn_R_48.1_38.3_-9.2
        roi_id, region, hemi = roi.split('_')[:3]
        full_region = region + '_' + hemi
        roi_num = int(roi_id[3:])
        if full_region in anat_to_roi:
            anat_to_roi[full_region].append(roi_num)
        else:
            anat_to_roi[full_region] = [roi_num]
    return anat_to_roi

def group_by_roi(subj_dict):
    examples_gordon, voxel_roi, rois_gordon = to_gordon(subj_dict)
    print(examples_gordon.shape)
    print(voxel_roi.shape)
    print(len(rois_gordon))
    anat_to_roi = roi_dict(rois_gordon)
    anat_to_examples = {}

    for anat, rois in anat_to_roi.items():
        print(anat)
        # get indices of columns that belong to rois associated with name
        voxels_from_rois = [i for i in range(len(voxel_roi)) if voxel_roi[i] in rois]
        # checked that these sum up to the number of columns in example_gordon
        examples_from_rois = examples_gordon[:, voxels_from_rois]
        print(examples_from_rois.shape)
        anat_to_examples[anat] = examples_from_rois
    return anat_to_examples