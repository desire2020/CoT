import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM, CUDNN_RNN_BIDIRECTION
from tensorflow.contrib import layers
class Critic(object):
    def __call__(self, h):
        # sequence -> [b, l, v]
        _, l, v = h.get_shape().as_list()
        h = tf.reshape(h, [-1, l, 1, v])
        with tf.variable_scope("textmover", reuse=tf.AUTO_REUSE):
            h0 = layers.convolution2d(
                h, v, [4, 1], [2, 1],
                activation_fn=tf.nn.softplus
            )
            h1 = layers.convolution2d(
                h0, v, [4, 1], [1, 1],
                activation_fn=tf.nn.softplus
            )
            h2 = layers.convolution2d(
                h1, v, [4, 1], [2, 1],
                activation_fn=tf.nn.softplus
            )
            h = layers.flatten(h2)
            h = layers.fully_connected(
                h, 1, activation_fn=tf.identity
            )
            return h

class Mediator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=1e-3, reward_gamma=0.95, name="mediator", dropout_rate=0.5, with_professor_forcing=False):
        self.num_emb = num_emb
        # self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.name = name
        self.dropout_keep_rate = tf.Variable(float(1.0), trainable=False)
        self.dropout_on = self.dropout_keep_rate.assign(dropout_rate)
        self.dropout_off = self.dropout_keep_rate.assign(1.0)
        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        self.x0 = tf.placeholder(tf.int32, shape=[None, self.sequence_length])
        self.x = self.x0
        self.x1 = tf.placeholder(tf.int32, shape=[None, self.sequence_length])
        input_x0 = tf.pad(self.x0, [[0, 0], [1, 0]])[:, 0:self.sequence_length]
        input_x1 = tf.pad(self.x1, [[0, 0], [1, 0]])[:, 0:self.sequence_length]
        output_x0 = tf.one_hot(
            self.x0, self.num_emb, on_value=1.0, off_value=0.0
        )
        output_x1 = tf.one_hot(
            self.x1, self.num_emb, on_value=1.0, off_value=0.0
        )
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(
                name="word_embeddings",
                initializer=tf.random_normal(shape=[self.num_emb, self.emb_dim], stddev=0.1)
            )
            Wo = tf.get_variable(
                name="Weight_output",
                initializer=tf.random_normal(shape=[self.hidden_dim, self.num_emb], stddev=0.1)
            )
            bo = tf.get_variable(
                name="bias_output",
                initializer=tf.random_normal(shape=[self.num_emb], stddev=0.1)
            )
            rnn = CudnnLSTM(
                num_layers=1,
                num_units=self.hidden_dim,
                kernel_initializer=tf.orthogonal_initializer()
            )
            def language_modeling(input_x):
                with tf.variable_scope("language_model", reuse=tf.AUTO_REUSE):
                    emb_x = tf.nn.embedding_lookup(
                        embedding, input_x
                    )
                    emb_x = tf.transpose(emb_x, [1, 0, 2])
                    h, _ = rnn(emb_x)
                    h = tf.transpose(h, [1, 0, 2])
                    h = tf.nn.dropout(h, self.dropout_keep_rate)
                    pred = tf.nn.log_softmax(
                        tf.reshape(h, [-1, self.hidden_dim]) @ Wo + bo,
                        axis=-1)
                    return h, tf.reshape(pred, [-1, self.sequence_length, self.num_emb])
            self.h0, self.log_predictions = language_modeling(input_x0)
            self.h1, self.log_predictions_ = language_modeling(input_x1)
            self.likelihood_loss = -tf.reduce_mean(
                tf.reduce_sum(
                    self.log_predictions * output_x0 +
                    self.log_predictions_ * output_x1, axis=-1)
            ) / 2.0
        self.m_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.95)
        if with_professor_forcing:
            with tf.variable_scope("professor_forcing", reuse=tf.AUTO_REUSE):
                critic = Critic()
                myu = tf.random_uniform(shape=[tf.shape(self.x0)[0], self.sequence_length, 1],
                                        minval=0.0, maxval=1.0)
                hybrid = self.h0 * myu + self.h1 * (1.0 - myu)
                gp = tf.reduce_mean(tf.nn.relu(tf.norm(
                    tf.reshape(tf.gradients(critic(hybrid), [hybrid])[0], [tf.shape(self.x0)[0], -1]),
                    axis=-1) - 1.0) ** 2)
                self.d_loss = tf.reduce_mean(critic(self.h0) - critic(self.h1))
                self.d_opt = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9)
                self.d_params = [v for v in tf.trainable_variables() if "professor_forcing" in v.name]
                self.d_update = self.d_opt.minimize(self.d_loss + 5.0 * gp, var_list=self.d_params)
        self.m_params = [v for v in tf.trainable_variables() if name in v.name]
        if not with_professor_forcing:
            self.likelihood_updates = self.m_opt.minimize(self.likelihood_loss, var_list=self.m_params)
        else:
            self.likelihood_updates = self.m_opt.minimize(self.likelihood_loss - self.d_loss, var_list=self.m_params)

    def get_reward(self, sess, x):
        output = sess.run(self.log_predictions, feed_dict={self.x0: x})
        return output
