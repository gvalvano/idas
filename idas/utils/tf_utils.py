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
import numpy as np


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


def reshape_tensor(tensor, dims_list):
    """ General purpose reshape function to collapse any list of dimensions.
        .Example
        We want to convert a Tensor of rank 3 to a tensor of rank 2 by collapsing the second and third dimensions
        into one:
          |  b = tf.placeholder(tf.float32, [None, 10, 32])
          |  shape = get_shape(b)
          |  b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
        With this function, we can easily write:
          |  b = tf.placeholder(tf.float32, [None, 10, 32])
          |  b = reshape(b, [0, [1, 2]])  # hence: collapse [1, 2] into the same dimension, leave 0 dimension unchanged
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
