"""
Utilities for matlab data
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

import scipy.io as sio


def get_matlab_matrix(filename, mdict=None, appendmat=True, **kwargs):
    """ Gets matlab matrix from a given path (filename). """
    return sio.loadmat(filename, mdict, appendmat, **kwargs)


def save_matlab_matrix(filename, mdict,
                       appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row'):
    """ Saves matlab matrix to given path (filename). """
    sio.savemat(filename, mdict, appendmat, format, long_field_names, do_compression, oned_as)


# path = '/Users/gabrielevalvano/Desktop/data_set/dati fMRI/deep_fags' + os.sep
# validation_size = 100
# 
# mat_contents = sio.loadmat(path + 'v1_s1_training.mat')
# xt = mat_contents['v1_s1_training'].transpose()[:-validation_size, :]
# xv = mat_contents['v1_s1_training'].transpose()[-validation_size:, :]
# 
# mat_contents = sio.loadmat(path + 'stim_training.mat')
# yt = mat_contents['stimTrn'][:-validation_size, :, :]
# yv = mat_contents['stimTrn'][-validation_size:, :, :]
