"""
Custom layer that rounds inputs during forward pass and copies the gradients during backward pass
"""
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

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def _py_function(pyfunc, incoming, out_types, stateful=True, name=None, grad=None):
    """
    Define custom py_func which takes also a grad op as argument:
    :param pyfunc: python function to perform
    :param incoming: list of `Tensor` objects.
    :param out_types: list or tuple of tensorflow data types
    :param stateful: (Boolean) If True, the function should be considered stateful.
    :param name: name scope
    :param grad: gradient policy
    :return:
    """
    # generate a unique name to avoid duplicates
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, int(1E+8)))
    tf.RegisterGradient(rnd_name)(grad)

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        res = tf.py_func(pyfunc, incoming, out_types, stateful=stateful, name=name)
        res[0].set_shape(incoming[0].get_shape())
        return res


def round_layer(incoming, name=None):
    """
    Rounds inputs during forward pass and copies the gradients during backward pass
    :param incoming: (tensor) input tensor
    :param name: (string) name scope (optional)
    :return:
    """
    with ops.name_scope(name, "RoundLayer", [incoming]) as name:
        round_incoming = _py_function(lambda x: np.round(x).astype('float32'), [incoming], [tf.float32], name=name,
                                      grad=lambda op, grad: grad)  # <-- here's the call to the gradient
        return round_incoming[0]
