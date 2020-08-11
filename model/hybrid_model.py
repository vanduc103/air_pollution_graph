from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lib import utils, metrics
from lib.metrics import masked_mae_loss
from model.hybrid_cell import HybridCell


class HybridModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        elif filter_type == "identity":
            supports.append(utils.calculate_identity_matrix(adj_mx))
            supports.append(utils.calculate_identity_matrix(adj_mx.T))
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, output_dim), name='labels')

        cell = HybridCell(rnn_units, num_nodes=num_nodes)
        lstm_cells = [cell] * num_rnn_layers
        lstm_cells = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('HYBRID_SEQ'):
            # Spatial layer (GCN)
            output_size = seq_len * input_dim
            g_outputs = self._gconv(self._inputs, output_size)
            # Temporal layer (LSTM)
            lstm_inputs = tf.reshape(g_outputs, [batch_size, seq_len, num_nodes, input_dim])
            lstm_inputs = tf.concat([lstm_inputs, self._inputs], axis=-1)
            lstm_inputs = tf.unstack(tf.reshape(lstm_inputs, (batch_size, seq_len, num_nodes * input_dim * 2)), axis=1)
            lstm_outputs, lstm_states = tf.contrib.rnn.static_rnn(lstm_cells, lstm_inputs, dtype=tf.float32)
            # Output layer (FNN)
            input_size = lstm_states[0].get_shape()[1].value
            output_size = horizon * num_nodes * output_dim
            weights = tf.get_variable(
                    'outputs_weights', [input_size, output_size], dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            value = tf.matmul(lstm_states[0], weights)
            biases = tf.get_variable("outputs_biases", [output_size], dtype=tf.float32,
                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            outputs = tf.nn.bias_add(value, biases)

        # Project the output to output_dim.
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _gconv(self, inputs, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                for support in self._supports:
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
