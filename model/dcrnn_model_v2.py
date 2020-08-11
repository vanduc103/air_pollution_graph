from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import legacy_seq2seq

from lib.metrics import masked_mae_loss
from model.dcrnn_cell import DCGRUCell


class DCRNNModel(object):
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
        input_dim_traffic = int(model_kwargs.get('input_dim_traffic', 1))
        input_dim_speed = int(model_kwargs.get('input_dim_speed', 1))
        input_dim_china = int(model_kwargs.get('input_dim_china', 1))
        input_dim_weather = int(model_kwargs.get('input_dim_weather', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        # Input (batch_size, timesteps, num_sensor, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        self._inputs_weather = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim_weather), name='inputs_weather')
        self._inputs_traffic = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim_traffic), name='inputs_traffic')
        self._inputs_speed = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim_speed), name='inputs_speed')
        self._inputs_china = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim_china), name='inputs_china')
        # Labels: (batch_size, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        # GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * input_dim))
        GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))

        cell_enc = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_weather = DCGRUCell(int(rnn_units/2), adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_traffic = DCGRUCell(int(rnn_units/2), adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_speed = DCGRUCell(int(rnn_units/2), adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_china = DCGRUCell(int(rnn_units/2), adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        encoding_cells = [cell_enc] * num_rnn_layers
        encoding_cells_weather = [cell_weather] * num_rnn_layers
        encoding_cells_traffic = [cell_traffic] * num_rnn_layers
        encoding_cells_speed = [cell_speed] * num_rnn_layers
        encoding_cells_china = [cell_china] * num_rnn_layers

        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells, state_is_tuple=True)
        encoding_cells_weather = tf.contrib.rnn.MultiRNNCell(encoding_cells_weather, state_is_tuple=True)
        encoding_cells_traffic = tf.contrib.rnn.MultiRNNCell(encoding_cells_traffic, state_is_tuple=True)
        encoding_cells_speed = tf.contrib.rnn.MultiRNNCell(encoding_cells_speed, state_is_tuple=True)
        encoding_cells_china = tf.contrib.rnn.MultiRNNCell(encoding_cells_china, state_is_tuple=True)

        cell_dec = DCGRUCell(rnn_units*3, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)
        cell_with_projection = DCGRUCell(rnn_units*3, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)
        decoding_cells = [cell_dec] * (num_rnn_layers - 1) + [cell_with_projection]
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)

        global_step = tf.train.get_or_create_global_step()
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        with tf.variable_scope('DCRNN_SEQ'):
            inputs = tf.unstack(tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim)), axis=1)
            inputs_weather = tf.unstack(tf.reshape(self._inputs_weather, (batch_size, seq_len, num_nodes * input_dim_weather)), axis=1)
            inputs_traffic = tf.unstack(tf.reshape(self._inputs_traffic, (batch_size, seq_len, num_nodes * input_dim_traffic)), axis=1)
            inputs_speed = tf.unstack(tf.reshape(self._inputs_speed, (batch_size, seq_len, num_nodes * input_dim_speed)), axis=1)
            inputs_china = tf.unstack(tf.reshape(self._inputs_china, (batch_size, seq_len, num_nodes * input_dim_china)), axis=1)

            labels = tf.unstack(
                tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim)), axis=1)
            labels.insert(0, GO_SYMBOL)

            def _loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    if use_curriculum_learning:
                        c = tf.random_uniform((), minval=0, maxval=1.)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        result = tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)
                    else:
                        result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                return result

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            _, enc_state_weather = tf.contrib.rnn.static_rnn(encoding_cells_weather, inputs_weather, dtype=tf.float32, scope="weather_enc")
            _, enc_state_traffic = tf.contrib.rnn.static_rnn(encoding_cells_traffic, inputs_traffic, dtype=tf.float32, scope="traffic_enc")
            _, enc_state_speed = tf.contrib.rnn.static_rnn(encoding_cells_speed, inputs_speed, dtype=tf.float32, scope="speed_enc")
            _, enc_state_china = tf.contrib.rnn.static_rnn(encoding_cells_china, inputs_china, dtype=tf.float32, scope="china_enc")
            concat_enc_state = (tf.concat([enc_state[0], enc_state_weather[0], enc_state_traffic[0], enc_state_speed[0], enc_state_china[0]], axis=1), \
                                tf.concat([enc_state[1], enc_state_weather[1], enc_state_traffic[1], enc_state_speed[1], enc_state_china[1]], axis=1))

            outputs, final_state = legacy_seq2seq.rnn_decoder(labels, concat_enc_state, decoding_cells,
                                                              loop_function=_loop_function)

        # Project the output to output_dim.
        outputs = tf.stack(outputs[:-1], axis=1)
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.summary.merge_all()

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
    def inputs_weather(self):
        return self._inputs_weather

    @property
    def inputs_traffic(self):
        return self._inputs_traffic

    @property
    def inputs_speed(self):
        return self._inputs_speed

    @property
    def inputs_china(self):
        return self._inputs_china

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
