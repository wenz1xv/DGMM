import numpy as np
import tensorflow as tf
import selfies as sf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model

# EMA class
from typing import Optional
# import sonnet as snt
# from sonnet.src.metrics import Metric
# from sonnet.src import types
# from sonnet.src import once

import pickle
import pandas as pd
from tensorflow.data import Dataset
import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
RDLogger.DisableLog("rdApp.*")

# @snt.allow_empty_variables
# class ExponentialMovingAverage(Metric):
#     """Maintains an exponential moving average for a value.

#     Note this module uses debiasing by default. If you don't want this please use
#     an alternative implementation.

#     This module keeps track of a hidden exponential moving average that is
#     initialized as a vector of zeros which is then normalized to give the average.
#     This gives us a moving average which isn't biased towards either zero or the
#     initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

#     Initially:
#       hidden_0 = 0
#     Then iteratively:
#       hidden_i = (hidden_{i-1} - value) * (1 - decay)
#       average_i = hidden_i / (1 - decay^i)
#     Attributes:
#     average: Variable holding average. Note that this is None until the first
#       value is passed.
#     """

#     def __init__(self, decay: types.FloatLike, name: Optional[str] = None):
#         """Creates a debiased moving average module.

#         Args:
#           decay: The decay to use. Note values close to 1 result in a slow decay
#             whereas values close to 0 result in faster decay, tracking the input
#             values more closely.
#           name: Name of the module.
#         """
#         super().__init__(name=name)
#         self._decay = decay
#         self._counter = tf.Variable(
#             0, trainable=False, dtype=tf.int64, name="counter")

#         self._hidden = None
#         self.average = None

#     def update(self, value: tf.Tensor):
#         """Applies EMA to the value given."""
#         self.initialize(value)
#         self._counter.assign_add(1)
#         value = tf.convert_to_tensor(value)
#         counter = tf.cast(self._counter, value.dtype)
#         self._hidden.assign_sub((self._hidden - value) * (1 - self._decay))
#         self.average.assign((self._hidden / (1. - tf.pow(self._decay, counter))))

#     @property
#     def value(self) -> tf.Tensor:
#         """Returns the current EMA."""
#         return self.average.read_value()

#     def reset(self):
#         """Resets the EMA."""
#         self._counter.assign(tf.zeros_like(self._counter))
#         self._hidden.assign(tf.zeros_like(self._hidden))
#         self.average.assign(tf.zeros_like(self.average))

#     @once.once
#     def initialize(self, value: tf.Tensor):
#         self._hidden = tf.Variable(
#             tf.zeros_like(value), trainable=False, name="hidden")
#         self.average = tf.Variable(
#             tf.zeros_like(value), trainable=False, name="average")

# class FSQ(layers.Layer):
#     def __init__(self, L, **kwargs):
#         super().__init__(**kwargs)
#         self.L = L
        
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "L": self.L
#         })
#         return config

#     def call(self, x):
#         trans_x = (self.L-1)*tf.sigmoid(x)
#         encoding_indices = tf.cast(tf.round(trans_x), tf.int32)
# #         encoding = tf.one_hot(encoding_indices, self.L)
#         encoding = trans_x + tf.stop_gradient(tf.round(trans_x) - trans_x)
#         e_mean = tf.cast(tf.reduce_mean(encoding_indices, axis=0), tf.dtypes.float64)
#         perplexity = tf.exp(-tf.reduce_sum(e_mean * tf.math.log(e_mean + 1e-10)))
#         return encoding_indices, perplexity
    
#     def get_code_indices(self, x):
#         trans_x = (self.L-1)*tf.sigmoid(x)
#         quantized = tf.round(trans_x)
#         return quantized

# class VectorQuantizer(layers.Layer):
#     def __init__(self, num_embeddings, embedding_dim, ema=True, decay=1e-3, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.L = num_embeddings
#         self.beta = 0.25 # between [0.25, 2] as per the paper
#         # Initialize the embeddings which we will quantize.
#         w_init = tf.random_normal_initializer()
#         self.epsilon=1e-5
#         self.embeddings = tf.Variable(
#             initial_value=w_init(
#                 shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
#             ),
#             trainable= not ema,
#             name="embeddings_vqvae",
#         )
#         self.ema = ema
#         if self.ema:
#             self.ema_cluster_size = ExponentialMovingAverage(decay=decay, name='ema_cluster_size')
#             self.ema_cluster_size.initialize(tf.zeros([num_embeddings], dtype=tf.float32))
#             self.ema_dw = ExponentialMovingAverage(decay=decay, name='ema_dw')
#             self.ema_dw.initialize(self.embeddings)
        
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "num_embeddings": self.num_embeddings,
#             "embedding_dim": self.embedding_dim,
#         })
#         return config

#     def call(self, x):
#         # Calculate the input shape of the inputs and
#         # then flatten the inputs keeping `embedding_dim` intact.
#         input_shape = tf.shape(x)
#         # Quantization.
#         encoding_indices = self.get_code_indices(x)
#         encodings = tf.one_hot(encoding_indices, self.num_embeddings)
#         quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
#         quantized = tf.reshape(quantized, input_shape)
        
#         commitment_loss = (tf.stop_gradient(quantized) - x)**2
#         commitment_loss = tf.reduce_mean(tf.reduce_sum(commitment_loss, axis=-1))
        
#         if self.ema:
#             flattened_inputs = tf.reshape(x, [-1, self.embedding_dim])
#             updated_ema_cluster_size = self.ema_cluster_size(tf.reduce_sum(encodings, axis=0))
#             dw = tf.matmul(flattened_inputs, encodings, transpose_a=True)
#             updated_ema_dw = self.ema_dw(dw)
#             n = tf.reduce_sum(updated_ema_cluster_size)
#             updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
#                                       (n + self.num_embeddings * self.epsilon) * n)
#             normalised_updated_ema_w = (updated_ema_dw / tf.reshape(updated_ema_cluster_size, [1, -1]))
#             self.embeddings.assign(normalised_updated_ema_w)
#             loss = self.beta * commitment_loss
#         else:
#             codebook_loss = (quantized - tf.stop_gradient(x))**2
#             codebook_loss = tf.reduce_mean(tf.reduce_sum(codebook_loss, axis=-1))
#             loss = self.beta * commitment_loss + codebook_loss
        
#         self.add_loss(loss)
        
#         e_mean = tf.reduce_mean(encodings, axis=0)
#         perplexity = tf.exp(-tf.reduce_sum(e_mean * tf.math.log(e_mean + 1e-10)))

#         # Straight-through estimator.
#         quantized = x + tf.stop_gradient(quantized - x)
#         return quantized, perplexity

#     def get_code_indices(self, x):
#         # Calculate L2-normalized distance between the inputs and the codes.
#         flattened_inputs = tf.reshape(x, [-1, self.embedding_dim])
#         similarity = tf.matmul(flattened_inputs, self.embeddings)
#         distances = (
#             tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
#             + tf.reduce_sum(self.embeddings ** 2, axis=0)
#             - 2 * similarity
#         )
#         encoding_indices = tf.argmin(distances, axis=1)
#         return encoding_indices


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z