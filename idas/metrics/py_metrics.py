import numpy as np


def eval_dice(seg, gt):
    """
    Returns Sørensen–Dice coefficient for binary masks (only 1 and 0).
    Be aware that the mask must only contain 0 and 1, otherwise you can end up with wrong scores.
    :param seg: segmentation mask
    :param gt: ground truth mask
    :return: dice score
    """
    dice = np.sum(seg[gt == 1])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice
