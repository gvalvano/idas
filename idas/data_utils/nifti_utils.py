"""
Utilities for nifti data
"""

import nibabel as nib
import numpy as np


def get_nifti_matrix(filename, dtype=np.int16):
    """ Returns array from nifti filename and affine matrix. """
    array = nib.load(filename).get_data().astype(dtype)  # array
    affine = nib.load(filename).affine  # affine matrix
    return array, affine


def save_nifti_matrix(array, affine, filename):
    """ Saves nifti array with a given affine matrix. """
    nimage = nib.Nifti1Image(array, affine)
    nib.save(nimage, filename=filename)
