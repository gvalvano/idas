import os
from tensorflow.python.client import device_lib
import tensorflow as tf


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def set_gpu_fraction(gpu_fraction=0.8):
    """ Set the GPU memory fraction for the application. 
    References: `TensorFlow using GPU <https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html>`
    :param gpu_fraction : (float) Fraction of GPU memory, (0 ~ 1]
    :return: -
    """
    tf.logging.info("[TL]: GPU MEM Fraction %f" % gpu_fraction)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return sess


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass


def print_yellow_text(text, sep=True):
    """ useful for debug """
    if sep:
        print('_' * 40)  # line separator
    print('\033[1;33m{0}:\033[0m'.format(text))


def create_gif(gif_name, path, ext='.png', delay=30, loop=0):
    """ Create gif from the list of images under the given path
    On Mac OS X, it requires "brew install ImageMagick".
    """
    cmd = 'convert -delay {0} -loop {1} {2}/*{3} {4}.gif'.format(delay, loop, path, ext, gif_name)
    os.system(cmd)
    print('gif successfully saved as {0}.gif'.format(os.path.join(path, gif_name)))
