#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf
from metrics import dice_coe, jaccard_coe, generalized_dice_coe, shannon_binary_entropy


def l2_weights_regularization_loss(exclude_bias=False):
    """
    Compute L2 regularization loss on all variable weights.

    Args:
        exclude_bias (Boolean): if True, exclude bias from loss computation.

    Returns:
        Global L2 loss term.

    """
    _vars = tf.trainable_variables()
    if exclude_bias:
        loss = tf.add_n([tf.nn.l2_loss(v) for v in _vars if 'bias' not in v.name])
    else:
        loss = tf.add_n([tf.nn.l2_loss(v) for v in _vars])
    return loss


def dice_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """
    Compute Soft Sørensen–Dice loss (also known as just DICE loss) for evaluating the similarity of two batch of
    data. The loss can vary between 0 and 1, where 1 means totally mismatch. It is usually used for binary image
    segmentation and is computed as: 1.0 - dice_coe(...).

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator while computing the Dice coefficient.

    Returns:
        Average Soft Dice loss loss on the batch.

    References:
        `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    return 1.0 - dice_coe(output, target, axis=axis, smooth=smooth)


def generalized_dice_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """
    Compute Generalized Soft Sørensen–Dice loss (also known as just Generalized DICE loss) for evaluating the similarity
    of two batch of data. The loss can vary between 0 and 1, where 1 means totally mismatch. It is usually used for
    binary image segmentation and is computed as: 1.0 - generalized_dice(...).

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator while computing the Dice coefficient.

    Returns:
        Average Generalized Dice loss loss on the batch.

    References:
        [1] Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss function for highly unbalanced
        segmentations." Deep learning in medical image analysis and multimodal learning for clinical decision support.
        Springer, Cham, 2017. 240-248.
    """
    return 1.0 - generalized_dice_coe(output, target, axis=axis, smooth=smooth)


def jaccard_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns Soft Jaccard (also known as Intersection over Union) loss for evaluating the similarity
    of two batch of data. The loss can vary between 0 and 1, where 1 means totally mismatch. It is usually used for
    binary image segmentation and is computed as: 1.0 - jaccard_coe(...).

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Average Jaccard loss on the batch.

    References:
        `Wiki-Jaccard <https://en.wikipedia.org/wiki/Jaccard_index>`__

    """
    return 1.0 - jaccard_coe(output, target, axis=axis, smooth=smooth)


def iou_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Wrapper to Soft Jaccard (also known as Intersection over Union) loss for evaluating the similarity
    of two batch of data. The loss can vary between 0 and 1, where 1 means totally mismatch. It is usually used for
    binary image segmentation and is computed as: 1.0 - jaccard_coe(...).

    Args:
        output (Tensor): a distribution with shape: [batch_size, ....], (any dimensions). This is the prediction.
        target (Tensor): the target distribution, format the same with `output`.
        axis (tuple of int): contains all the dimensions to be reduced, default ``[1,2,3]``.
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Average Intersection over Union loss on the batch.

    References:
        `Wiki-Jaccard <https://en.wikipedia.org/wiki/Jaccard_index>`__
    """
    return 1.0 - jaccard_coe(output, target, axis=axis, smooth=smooth, _name='iou_coe')


def weighted_softmax_cross_entropy(y_pred, y_true, num_classes, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks. Applies a softmax operation to y_pred to give it a
    probabilistic interpretation.

    Args:
        y_pred (tensor): predicted tensor [None, width, height, n_classes]
        y_true (tensor): target tensor [None, width, height, n_classes]
        num_classes (int): number of classes
        eps (float): small value to avoid division by zero

    Returns:
        Average weighted cross-entropy loss on the batch.
    """

    n = [tf.reduce_sum(tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(n)

    weights = [n_tot / (n[c] + eps) for c in range(num_classes)]

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))
    softmax = tf.nn.softmax(y_pred)

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax + eps), weights), reduction_indices=[1])
    loss = tf.reduce_mean(w_cross_entropy, name='weighted_softmax_cross_entropy')
    return loss


def weighted_cross_entropy(y_pred, y_true, num_classes, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks. Assumes y_pred already probabilistic.

    Args:
        y_pred (tensor): predicted tensor [None, width, height, n_classes]
        y_true (tensor): target tensor [None, width, height, n_classes]
        num_classes (int): number of classes
        eps (float): small value to avoid division by zero

    Returns:
        Average weighted cross-entropy loss on the batch.
    """

    n = [tf.reduce_sum(tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(n)

    weights = [n_tot / (n[c] + eps) for c in range(num_classes)]

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred + eps), weights), reduction_indices=[1])
    loss = tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')
    return loss


def segmentation_weighted_cross_entropy(y_pred, y_true, n_voxels, eps=1e-12):
    """
    Compute weighted cross-entropy on binary segmentation masks.

    Args:
        y_pred (tensor): predicted tensor [None, width, height, n_classes]
        y_true (tensor): target tensor [None, width, height, n_classes]
        n_voxels (): number of voxels (i.e. N * M)
        eps (float): small value to avoid division by zero

    Returns:
        Average weighted cross-entropy loss on the batch.

    """

    a = tf.divide(1., tf.reduce_sum(tf.cast(y_true, tf.float32)))
    b = tf.divide(1., (n_voxels - a))
    weights = [b, a]  # [1/(number of zeros), 1/(number of ones)]

    num_classes = y_pred.get_shape().as_list()[-1]  # class on the last index
    assert (num_classes is not None)
    assert len(weights) == num_classes

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))
    softmax = tf.nn.softmax(y_pred) + eps

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax), weights), reduction_indices=[1])

    return tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')


def vae_loss(z_mean, z_logstd, y_true, y_pred, n_outputs):
    """
    Computes classical loss for variational auto-encoders, as:
        Loss = Reconstruction_loss + KL_divergence_loss
             = MSE - ELBO

    Args:
        z_mean (tensor): encoded mean in the VAE latent vector.
        z_logstd (tensor): encoded logarithm of the standard deviation in the VAE latent vector.
        y_true (tensor): true input sample to the VAE
        y_pred (tensor): reconstruction of the correspondent sampled sample in the VAE latent code.
        n_outputs (int): number of outputs (i.e. in mnist we have 28x28 = 784)

    Returns:
        VAE loss computed as MSE on the reconstruction - ELBO

    """
    x_true = tf.reshape(y_true, shape=(-1, n_outputs))
    x_reconstructed = tf.reshape(y_pred, shape=(-1, n_outputs))

    # _______
    # Reconstruction loss:
    with tf.variable_scope('Reconstruction_loss'):
        generator_loss = tf.losses.mean_squared_error(x_reconstructed, x_true)

    # _______
    # KL Divergence loss:
    with tf.variable_scope('KL_divergence_loss'):
        kl_div_loss = 1.0 + z_logstd - tf.square(z_mean) - tf.exp(z_logstd)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    total_loss = tf.reduce_mean(generator_loss + kl_div_loss)

    return total_loss, generator_loss, kl_div_loss


def gradient_loss(y_pred, y_true, shape):
    """
    Computes gradient loss as the mean square error on the image filtered with Sobel filters. In particular, it computes
    the gradients of both the predicted and the target image and computes the MSE between the two.

    Args:
        y_pred (tensor): predicted image (assumed to have continuous values).
        y_true (tensor): true image (assumed to have continuous values).
        shape (tuple of int): width and height of the image.

    Returns:
        MSE between the gradient of the predicted and true image.

    """
    assert len(shape) == 2

    pred_grad = tf.reduce_sum(tf.image.sobel_edges(y_pred[:, :, :, 0]), -1)
    input_grad = tf.reduce_sum(tf.image.sobel_edges(y_true[:, :, :, 0]), -1)
    
    x_reconstructed_grad = tf.reshape(pred_grad, shape=(-1, shape[0]*shape[1]))
    x_true_grad = tf.reshape(input_grad, shape=(-1, shape[0]*shape[1]))

    grad_loss = tf.reduce_mean(tf.squared_difference(x_reconstructed_grad, x_true_grad), 1)
    return grad_loss


def shannon_binary_entropy_loss(incoming, axis=(1, 2), unscaled=True, smooth=1e-12):
    """
    Evaluates shannon entropy loss on a binary mask. The last index of the incoming tensor must contain the one-hot
    encoded predictions.

    Args:
        incoming (tensor): incoming tensor (one-hot encoded). On the first dimension there is the number of samples
            (typically the batch size)
        axis (tuple of int): axis containing the input dimension. Assuming 'incoming' to be a 4D tensor, axis has length
            2: width and height; if 'incoming' is a 5D tensor, axis should have length of 3, and so on.
        unscaled (Boolean): The computation does the operations using the natural logarithm log(). To obtain the actual
            entropy alue one must scale this value by log(2) since the entropy should be computed in base 2 (hence
            log2(.)). However, one may desire to use this function in a loss function to train a neural net. Then, the
            log(2) is just a multiplicative constant of the gradient and could be omitted for efficiency reasons.
            Turning this flag to True allows for this behaviour to happen (default is False, then the actual entropy).
        smooth (float): small value added to the numerator and denominator.

    Returns:
        Entropy value of the incoming tensor.

    """
    return shannon_binary_entropy(incoming, axis=axis, unscaled=unscaled, smooth=smooth)
