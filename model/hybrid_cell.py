from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from lib import utils


class HybridCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, num_nodes, num_proj=None,
                 activation=tf.nn.tanh, reuse=None):
        """

        :param num_units:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        """
        super(HybridCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units


    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "gru_cell"):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                output_size = 2 * self._num_units
                # We start with bias of 1.0 to not reset and not update.
                value = tf.nn.sigmoid(self._fc(inputs, state, output_size, bias_start=1.0))
                value = tf.reshape(value, (-1, self._num_nodes, output_size))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))
            with tf.variable_scope("candidate"):
                c = self._fc(inputs, r * state, self._num_units)
                c = tf.reshape(c, (-1, self._num_nodes * self._num_units))
                if self._activation is not None:
                    c = self._activation(c)
            output = new_state = u * state + (1 - u) * c
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

