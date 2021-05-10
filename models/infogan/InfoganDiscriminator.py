import tensorflow as tf
from functools import partial

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.compat.v1.variable_scope(scope or "SimpleLinear"):
        matrix = tf.compat.v1.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.compat.v1.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.compat.v1.matmul(input_, tf.compat.v1.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.compat.v1.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.compat.v1.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.compat.v1.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            emd_dim, filter_sizes, num_filters, l2_reg_lambda=0.0, dropout_keep_prob = 1):
        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = dropout_keep_prob
        # self.dropout_keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.compat.v1.constant(0.0)

        # Convolution Layer - line 96
        filter_shape = [filter_size, emd_dim, 1, num_filter]
        W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_filter]), name="b")
        conv = tf.compat.v1.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        # Apply nonlinearity
        h = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, b), name="relu")

        #LeakyReLu - line 97
        output = tf.compat.v1.layers.dense(input, n_units, activation=partial(tf.nn.leaky_relu, alpha=0.01))

        # Add dropout - line 98
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.compat.v1.nn.dropout(self.h_highway, self.dropout_keep_prob)


        # Convolution Layer - line 99
        filter_shape = [filter_size, emd_dim, 1, num_filter]
        W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_filter]), name="b")
        conv = tf.compat.v1.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, b), name="relu")


        # line 100
        tf.compat.v1.pad(
            tensor, paddings, mode='CONSTANT', name=None, constant_values=0
        )


        #LeakyReLu - line 101
        output = tf.compat.v1.layers.dense(input, n_units, activation=partial(tf.nn.leaky_relu, alpha=0.01))

        # Add dropout - line 102
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.compat.v1.nn.dropout(self.h_highway, self.dropout_keep_prob)

        #Batch Normalization - line 103
        tf.compat.v1.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(), beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
            renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None,
            trainable=True, virtual_batch_size=None, adjustment=None, name=None, **kwargs
        )

        # Convolution Layer - line 104
        filter_shape = [filter_size, emd_dim, 1, num_filter]
        W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_filter]), name="b")
        conv = tf.compat.v1.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, b), name="relu")

         #LeakyReLu - line 105
        output = tf.compat.v1.layers.dense(input, n_units, activation=partial(tf.nn.leaky_relu, alpha=0.01))

        # Add dropout - line 106
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.compat.v1.nn.dropout(self.h_highway, self.dropout_keep_prob)


        #Batch Normalization - line 107
        tf.compat.v1.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(), beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
            renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None,
            trainable=True, virtual_batch_size=None, adjustment=None, name=None, **kwargs
        )

        # Convolution Layer - line 108
        filter_shape = [filter_size, emd_dim, 1, num_filter]
        W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_filter]), name="b")
        conv = tf.compat.v1.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, b), name="relu")

        #LeakyReLu - line 109
        output = tf.compat.v1.layers.dense(input, n_units, activation=partial(tf.nn.leaky_relu, alpha=0.01))
        

        # Add dropout - line 110
        with tf.compat.v1.name_scope("dropout"):
            self.h_drop = tf.compat.v1.nn.dropout(self.h_highway, self.dropout_keep_prob)


        #Batch Normalization - line 111
        tf.compat.v1.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(), beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
            renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None,
            trainable=True, virtual_batch_size=None, adjustment=None, name=None, **kwargs
        )

        #flatten - line 112
        tf.nest.flatten(
            structure, expand_composites=False
        )

        # CalculateMean cross-entropy loss
        with tf.compat.v1.name_scope("loss"):
            losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.compat.v1.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.d_loss = tf.compat.v1.reshape(tf.compat.v1.reduce_mean(self.loss), shape=[1])

    self.params = [param for param in tf.compat.v1.trainable_variables() if 'discriminator' in param.name]
    d_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
    self.train_op = d_optimizer.apply_gradients(grads_and_vars)
