import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np


class Generator(object):
    def __init__(self, num_vocabulary, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.01, reward_gamma=0.95):
        self.num_vocabulary = num_vocabulary
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.compat.v1.constant([start_token] * self.batch_size, dtype=tf.compat.v1.int32)
        self.learning_rate = tf.compat.v1.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0

        self.expected_reward = tf.compat.v1.Variable(tf.compat.v1.zeros([self.sequence_length]))

        with tf.compat.v1.variable_scope('generator'):
          #add layers
            self.g_embeddings = tf.compat.v1.Variable(self.init_matrix([self.num_vocabulary, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

        # placeholder definition
        self.x = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[self.batch_size,
                                                 self.sequence_length])  # sequence of tokens generated by generator
        self.rewards = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[self.batch_size,
                                                         self.sequence_length])  # get from rollout policy and discriminator

        # processed for batch
        with tf.compat.v1.device("/cpu:0"):
            self.processed_x = tf.compat.v1.transpose(tf.compat.v1.nn.embedding_lookup(self.g_embeddings, self.x),
                                            perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        # Initial states
        self.h0 = tf.compat.v1.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.compat.v1.stack([self.h0, self.h0])

        gen_o = tensor_array_ops.TensorArray(dtype=tf.compat.v1.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.compat.v1.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.compat.v1.log(tf.compat.v1.nn.softmax(o_t))
            next_token = tf.compat.v1.cast(tf.compat.v1.reshape(tf.compat.v1.multinomial(log_prob, 1), [self.batch_size]), tf.compat.v1.int32)
            x_tp1 = tf.compat.v1.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_o = gen_o.write(i, tf.compat.v1.reduce_sum(tf.compat.v1.multiply(tf.compat.v1.one_hot(next_token, self.num_vocabulary, 1.0, 0.0),
                                                             tf.compat.v1.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.compat.v1.constant(0, dtype=tf.compat.v1.int32),
                       tf.compat.v1.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.compat.v1.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.compat.v1.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.compat.v1.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.compat.v1.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.compat.v1.constant(0, dtype=tf.compat.v1.int32),
                       tf.compat.v1.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        self.g_predictions = tf.compat.v1.transpose(self.g_predictions.stack(),
                                          perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        # pretraining loss
        self.pretrain_loss = -tf.compat.v1.reduce_sum(
            tf.compat.v1.one_hot(tf.compat.v1.to_int32(tf.compat.v1.reshape(self.x, [-1])), self.num_vocabulary, 1.0, 0.0) * tf.compat.v1.log(
                tf.compat.v1.clip_by_value(tf.compat.v1.reshape(self.g_predictions, [-1, self.num_vocabulary]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        # training updates
        pretrain_opt = self.g_optimizer(self.learning_rate)

        self.pretrain_grad, _ = tf.compat.v1.clip_by_global_norm(tf.compat.v1.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

        #######################################################################################################
        #  Unsupervised Training
        #######################################################################################################
        self.g_loss = -tf.compat.v1.reduce_sum(
            tf.compat.v1.reduce_sum(
                tf.compat.v1.one_hot(tf.compat.v1.to_int32(tf.compat.v1.reshape(self.x, [-1])), self.num_vocabulary, 1.0, 0.0) * tf.compat.v1.log(
                    tf.compat.v1.clip_by_value(tf.compat.v1.reshape(self.g_predictions, [-1, self.num_vocabulary]), 1e-20, 1.0)
                ), 1) * tf.compat.v1.reshape(self.rewards, [-1])
        )

        g_opt = self.g_optimizer(self.learning_rate)

        self.g_grad, _ = tf.compat.v1.clip_by_global_norm(tf.compat.v1.gradients(self.g_loss, self.g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

    def generate(self, sess):
        outputs = sess.run(self.gen_x)
        return outputs

    def pretrain_step(self, sess, x):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.x: x})
        return outputs

    def init_matrix(self, shape):
        return tf.compat.v1.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.compat.v1.zeros(shape)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.compat.v1.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.compat.v1.sigmoid(
                tf.compat.v1.matmul(x, self.Wi) +
                tf.compat.v1.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.compat.v1.sigmoid(
                tf.compat.v1.matmul(x, self.Wf) +
                tf.compat.v1.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.compat.v1.sigmoid(
                tf.compat.v1.matmul(x, self.Wog) +
                tf.compat.v1.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.compat.v1.nn.tanh(
                tf.compat.v1.matmul(x, self.Wc) +
                tf.compat.v1.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.compat.v1.nn.tanh(c)

            return tf.compat.v1.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.num_vocabulary]))
        self.bo = tf.compat.v1.Variable(self.init_matrix([self.num_vocabulary]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.compat.v1.unstack(hidden_memory_tuple)
            logits = tf.compat.v1.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        return tf.compat.v1.train.AdamOptimizer(*args, **kwargs)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:

    def set_similarity(self, valid_examples=None, pca=True):
        if valid_examples == None:
            if pca:
                valid_examples = np.array(range(20))
            else:
                valid_examples = np.array(range(self.num_vocabulary))
        self.valid_dataset = tf.compat.v1.constant(valid_examples, dtype=tf.compat.v1.int32)
        self.norm = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(self.g_embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.g_embeddings / self.norm
        # PCA
        if self.num_vocabulary >= 20 and pca == True:
            emb = tf.compat.v1.matmul(self.normalized_embeddings, tf.compat.v1.transpose(self.normalized_embeddings))
            s, u, v = tf.compat.v1.svd(emb)
            u_r = tf.compat.v1.strided_slice(u, begin=[0, 0], end=[20, self.num_vocabulary], strides=[1, 1])
            self.normalized_embeddings = tf.compat.v1.matmul(u_r, self.normalized_embeddings)
        self.valid_embeddings = tf.compat.v1.nn.embedding_lookup(
            self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.compat.v1.matmul(self.valid_embeddings, tf.compat.v1.transpose(self.normalized_embeddings))
