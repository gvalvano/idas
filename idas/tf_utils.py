import tensorflow as tf


def get_shape(tensor):
    """ It returns the static shape of a tensor when available, otherwise returns its dynamic shape.
        .Example
          |  a.set_shape([32, 128])  # static shape of a is [32, 128]
          |  a.set_shape([None, 128])  # first dimension of a is determined dynamicall
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def reshape(tensor, dims_list):
    """ General purpose reshape function to collapse any list of dimensions
    """
    shape = get_shape(tensor)
    dims_prod = []
    for dims in dims_list:
        if isinstance(dims, int):
            dims_prod.append(shape[dims])
        elif all([isinstance(shape[d], int) for d in dims]):
            dims_prod.append(np.prod([shape[d] for d in dims]))
        else:
            dims_prod.append(tf.reduce_prod([shape[d] for d in dims]))
    tensor = tf.reshape(tensor, dims_prod)
    return tensor
