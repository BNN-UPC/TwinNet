"""
   Copyright 2021 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import print_function
import tensorflow as tf
import time


class GNN_Model(tf.keras.Model):
    """ Init method for the custom model.

    Args:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        output_units (int): Output units for the last readout's layer.

    Attributes:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        link_update (GRUCell): Link GRU Cell used in the Message Passing step.
        path_update (GRUCell): Path GRU Cell used in the Message Passing step.
        queue_update (GRUCell): Queue GRU Cell used in the Message Passing step.
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(GNN_Model, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['path_state_dim']))
        self.link_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.queue_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['queue_state_dim']))

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['path_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2'])),
                                  ),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2']))),
            tf.keras.layers.Dense(output_units,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2_2'])))
        ])

    @tf.function
    def call(self, inputs, training=False):
        """This function is execution each time the model is called

        Args:
            inputs (dict): Features used to make the predictions.
            training (bool): Whether the model is train or not. If False, the
                             model does not update the weights.

        Returns:
            tensor: A tensor containing the per-path delay.
        """

        # Compute the shape for the  all-zero tensor for link_state
        path_shape = tf.stack([
            inputs['n_paths'],
            int(self.config['HYPERPARAMETERS']['link_state_dim']) -
            2
        ], axis=0)

        # Initialize the initial hidden state for links
        path_state = tf.concat([
            tf.expand_dims(inputs['traffic'], axis=1),
            tf.expand_dims(inputs['packets'], axis=1),
            tf.zeros(path_shape)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for path_state
        link_shape = tf.stack([
            inputs['n_links'],
            int(self.config['HYPERPARAMETERS']['path_state_dim']) -
            int(self.config['DATASET']['num_policies']) -
            1
        ], axis=0)

        # Initialize the initial hidden state for paths
        link_state = tf.concat([
            tf.expand_dims(inputs['capacity'], axis=1),
            tf.one_hot(inputs['policy'], int(self.config['DATASET']['num_policies'])),
            tf.zeros(link_shape)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for path_state
        queue_shape = tf.stack([
            inputs['n_queues'],
            int(self.config['HYPERPARAMETERS']['path_state_dim']) -
            int(self.config['DATASET']['max_num_queues']) -
            2
        ], axis=0)

        # Initialize the initial hidden state for paths
        queue_state = tf.concat([
            tf.expand_dims(inputs['size'], axis=1),
            tf.one_hot(inputs['priority'], int(self.config['DATASET']['max_num_queues'])),
            tf.expand_dims(inputs['weight'], axis=1),
            tf.zeros(queue_shape)
        ], axis=1)

        # Iterate t times doing the message passing
        for it in range(int(self.config['HYPERPARAMETERS']['t'])):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            link_gather = tf.gather(link_state, inputs['link_to_path'])
            queue_gather = tf.gather(queue_state, inputs['queue_to_path'])

            ids_q = tf.stack([inputs['path_ids'], inputs['l_q_p']], axis=1)
            ids_l = tf.stack([inputs['path_ids'], inputs['l_p_s']], axis=1)

            max_len = tf.reduce_max(inputs['l_q_p']) + 1
            shape = tf.stack([inputs['n_paths'], max_len, int(self.config['HYPERPARAMETERS']['path_state_dim'])])

            queue_input = tf.scatter_nd(ids_q, queue_gather, shape)
            link_input = tf.scatter_nd(ids_l, link_gather, shape)

            lens = tf.math.segment_sum(data=tf.ones_like(inputs['path_ids']),
                                       segment_ids=inputs['path_ids'])

            path_gru_rnn = tf.keras.layers.RNN(self.path_update, return_sequences=False)

            path_state = path_gru_rnn(inputs=tf.concat([queue_input, link_input], axis=2),
                                      initial_state=path_state,
                                      mask=tf.sequence_mask(lens))

            ###################
            #  PATH TO QUEUE  #
            ###################
            path_gather = tf.gather(path_state, inputs['path_to_queue'])
            path_sum = tf.math.unsorted_segment_sum(path_gather, inputs['sequence_queues'], inputs['n_queues'])
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            queue_gather = tf.gather(queue_state, inputs['queue_to_link'])
            ids_q = tf.stack([inputs['sequence_links'], inputs['l_q_l']], axis=1)
            max_len = tf.reduce_max(inputs['l_q_l']) + 1
            shape = tf.stack([inputs['n_links'], max_len, int(self.config['HYPERPARAMETERS']['link_state_dim'])])
            queue_input = tf.scatter_nd(ids_q, queue_gather, shape)

            lens = tf.math.segment_sum(data=tf.ones_like(inputs['sequence_links']),
                                       segment_ids=inputs['sequence_links'])

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(inputs=queue_input, initial_state=link_state, mask=tf.sequence_mask(lens))

        # Call the readout ANN and return its predictions
        r = self.readout(path_state, training=training)

        return r


def r_squared(labels, predictions):
    """Computes the R^2 score.

        Args:
            labels (tf.Tensor): True values
            labels (tf.Tensor): This is the second item returned from the input_fn passed to train, evaluate, and predict.
                                If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed.

        Returns:
            tf.Tensor: Mean R^2
        """

    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    r_sq = 1.0 - tf.truediv(unexplained_error, total_error)

    # Needed for tf2 compatibility.
    m_r_sq, update_rsq_op = tf.compat.v1.metrics.mean(r_sq)

    return m_r_sq, update_rsq_op


def model_fn(features, labels, mode, params):
    """model_fn used by the estimator, which, given inputs and a number of other parameters,
       returns the ops necessary to perform train, evaluation, or predictions.

    Args:
        features (dict): This is the first item returned from the input_fn passed to train, evaluate, and predict.
        labels (tf.Tensor): This is the second item returned from the input_fn passed to train, evaluate, and predict.
                            If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed.
        mode (tf.estimator.ModeKeys): Specifies if this is train, evaluation or prediction.
        params (dict): Dict of hyperparameters. Will receive what is passed to Estimator in params parameter.

    Returns:
        tf.estimator.EstimatorSpec: Ops and objects returned from a model_fn and passed to an Estimator.
    """

    # Create the model.
    model = GNN_Model(params)

    # start_time = tf.timestamp()
    # Execute the call function and obtain the predictions.
    predictions = model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = tf.squeeze(predictions)

    # If we are performing predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predicted values.
        return tf.estimator.EstimatorSpec(
            mode, predictions={
                'predictions': predictions
            })

    # Define the loss function.
    loss_function = tf.keras.losses.MeanSquaredError()

    # Obtain the regularization loss of the model.
    regularization_loss = sum(model.losses)

    # Compute the loss defined previously.
    loss = loss_function(labels, predictions)

    # Compute the total loss.
    total_loss = loss + regularization_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)

    # If we are performing evaluation.
    if mode == tf.estimator.ModeKeys.EVAL:
        denormalized_labels = tf.math.exp(labels)
        denormalized_predictions = tf.math.exp(predictions)
        # Define the different eval metrics
        label_mean = tf.keras.metrics.Mean()
        _ = label_mean.update_state(labels)
        prediction_mean = tf.keras.metrics.Mean()
        _ = prediction_mean.update_state(predictions)
        mae = tf.keras.metrics.MeanAbsoluteError()
        _ = mae.update_state(labels, predictions)
        mre = tf.keras.metrics.MeanRelativeError(normalizer=tf.abs(labels))
        _ = mre.update_state(labels, predictions)
        # Check the eval metrics after denormalizing
        denorm_label_mean = tf.keras.metrics.Mean()
        _ = denorm_label_mean.update_state(denormalized_labels)
        denorm_prediction_mean = tf.keras.metrics.Mean()
        _ = denorm_prediction_mean.update_state(denormalized_predictions)
        denorm_mae = tf.keras.metrics.MeanAbsoluteError()
        _ = denorm_mae.update_state(denormalized_labels, denormalized_predictions)
        denorm_mre = tf.keras.metrics.MeanRelativeError(normalizer=denormalized_labels)
        _ = denorm_mre.update_state(denormalized_labels, denormalized_predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': label_mean,
                'prediction/mean': prediction_mean,
                'mae': mae,
                'mre': mre,
                'r-squared': r_squared(labels, predictions),
                'denorm_label/mean': denorm_label_mean,
                'denorm_prediction/mean': denorm_prediction_mean,
                'denorm_mae': denorm_mae,
                'denorm_mre': denorm_mre,
                'denorm_r_squared': r_squared(denormalized_labels, denormalized_predictions)
            }
        )

    # If we are performing train.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Compute the gradients.
    grads = tf.gradients(total_loss, model.trainable_variables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in model.trainable_variables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    # Define an exponential decay schedule.
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(params['HYPERPARAMETERS']['learning_rate']),
                                                                int(params['HYPERPARAMETERS']['decay_steps']),
                                                                float(params['HYPERPARAMETERS']['decay_rate']),
                                                                staircase=True)

    # Define an Adam optimizer using the defined exponential decay.
    optimizer = tf.keras.optimizers.Adam(learning_rate=decayed_lr)

    # Manually assign tf.compat.v1.global_step variable to optimizer.iterations
    # to make tf.compat.v1.train.global_step increased correctly.
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    # Apply the processed gradients using the optimizer.
    train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Define the logging hook. It returns the loss, the regularization loss and the
    # total loss every 10 iterations.
    logging_hook = tf.estimator.LoggingTensorHook(
        {"Loss": loss,
         "Regularization loss": regularization_loss,
         "Total loss": total_loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )
