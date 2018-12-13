import tensorflow as tf


def weighted_cross_entropy(labels, logits):
    """
    define weighted cross-entropy function for classification tasks.
    1. In binary classification, each output channel corresponds to a binary (soft) decision. Therefore, the
    weighting needs to happen within the computation of the loss --> 'weighted_cross_entropy_with_logits'.
    2. In mutually exclusive multilabel classification each output channel corresponds to the score of a class
    candidate. The decision comes after and then --> 'softmax_cross_entropy_with_logits'
    """
    epsilon = tf.constant(1e-12, dtype=logits.dtype)

    # specify class weights:
    class_weights = tf.divide(1., tf.reduce_sum(tf.cast(labels, tf.float32), axis=0) + epsilon)

    # assign a weight for each sample inside the current minibatch
    weights = tf.reduce_sum(tf.multiply(tf.cast(labels, tf.float32), class_weights), 1)

    epsilon = tf.constant(1e-12, dtype=logits.dtype)
    print("Loss function: 'weighted cross_entropy', weights evaluated on minibatch samples")

    num_classes = labels.get_shape().as_list()[-1]  # classes are on the last index

    y_pred = tf.reshape(logits, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(labels, (-1, num_classes)))
    softmax = tf.nn.softmax(y_pred) + epsilon

    cross_entropy = -tf.reduce_sum(tf.multiply(y_true, tf.log(softmax)), reduction_indices=[1])
    w_cross_entropy = tf.multiply(cross_entropy, weights)

    return tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')


def segmentation_weighted_cross_entropy(y_pred, y_true, n_voxels, epsilon=1e-12):
    """
    Compute weighted cross-entropy on binary segmentation masks.
    :param y_pred: predicted output (i.e. with shape [None, N, M, num_classes]
    :param y_true: true output mask (i.e. with shape [None, N, M, num_classes]
    :param n_voxels: number of voxels (i.e. N * M)
    :param epsilon: constant to prevent division from 0
    :return: mean cross-entropy loss
    """

    a = tf.divide(1., tf.reduce_sum(tf.cast(y_true, tf.float32)))
    b = tf.divide(1., (n_voxels - a))
    weights = [b, a]  # [1/(number of zeros), 1/(number of ones)]

    num_classes = y_pred.get_shape().as_list()[-1]  # class on the last index
    assert (num_classes is not None)
    assert len(weights) == num_classes

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))
    softmax = tf.nn.softmax(y_pred) + epsilon

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax), weights), reduction_indices=[1])

    return tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')


def vae_loss(z_mean, z_logstd, y_true, y_pred, n_outputs):
    """
    Computes classical loss for variational auto-encoders, as:
      Loss = Reconstruction_loss + KL_divergence_loss
           = MSE - ELBO
    :param z_mean: encoded z_mean vector
    :param z_logstd: encoded z_logstd vector
    :param n_outputs: (i.e. in mnist we have 28x28 = 784)
    :return:
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
