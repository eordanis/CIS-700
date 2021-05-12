import tensorflow as tf


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
        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_classes], name="input_y")
        # self.input_x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=(None, sequence_length, sequence_length, 10), name="input_x")
        # self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=(None, sequence_length, sequence_length, 10), name="input_y")
        # x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=(None, sequence_length, sequence_length, 1), name="x")
        # z = tf.compat.v1.placeholder(tf.float32, shape=(None, sequence_length), name="input_z")
        # y_label = tf.compat.v1.placeholder(tf.float32, shape=(None, 1, 1, 10))
        # y_fill = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=(None, sequence_length, sequence_length, 10), name="y_fill")
        self.dropout_keep_prob = dropout_keep_prob
        # self.dropout_keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32, name="dropout_keep_prob")
        num_filters_total = sum(num_filters)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.compat.v1.constant(0.0)

        with tf.compat.v1.variable_scope('discriminator'):

            # dcgan initializer
            w_init = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.compat.v1.constant_initializer(0.0)

            # self.input_x = tf.compat.v1.reshape(self.input_x, (None, sequence_length, sequence_length, 10))
            # concat layer
            self.input_x_expanded = tf.compat.v1.expand_dims(self.input_x, -1)
            self.input_y_expanded = tf.compat.v1.expand_dims(self.input_y, -1)
            self.input_x_expanded = tf.compat.v1.expand_dims(self.input_x_expanded, -1)
            self.input_y_expanded = tf.compat.v1.expand_dims(self.input_y_expanded, -1)
            cat1 = tf.compat.v1.concat([self.input_x_expanded, self.input_y_expanded], 1)

            # 1st hidden layer
            conv1 = tf.compat.v1.layers.conv2d(cat1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            lrelu1 = tf.nn.leaky_relu(conv1, 0.2)

            # 2nd hidden layer
            conv2 = tf.compat.v1.layers.conv2d(lrelu1, 300, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
            lrelu2 = tf.nn.leaky_relu(tf.compat.v1.layers.batch_normalization(conv2, training=False), 0.2)

            # output layer
            # conv3 = tf.compat.v1.layers.conv2d(lrelu2, 300, [5, 5], strides=(1, 1), padding='valid', kernel_initializer=w_init)
            o = tf.nn.sigmoid(conv2)

            # Final (unnormalized) scores and predictions
            with tf.compat.v1.name_scope("output"):
                W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.compat.v1.nn.l2_loss(W)
                l2_loss += tf.compat.v1.nn.l2_loss(b)
                self.scores = tf.compat.v1.nn.xw_plus_b(o, W, b, name="scores")
                self.ypred_for_auc = tf.compat.v1.nn.softmax(self.scores)
                self.predictions = tf.compat.v1.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.compat.v1.name_scope("loss"):
                losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.compat.v1.reduce_mean(losses) + l2_reg_lambda * l2_loss
                self.d_loss = tf.compat.v1.reshape(tf.compat.v1.reduce_mean(self.loss), shape=[1])

            self.params = [param for param in tf.compat.v1.trainable_variables() if 'discriminator' in param.name]
            d_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
            grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
            self.train_op = d_optimizer.apply_gradients(grads_and_vars)
