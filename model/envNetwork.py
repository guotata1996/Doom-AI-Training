import tensorflow as tf
import time
from baselines.a2c.utils import *
from model.util import norm_factor
import joblib
from tensorflow.contrib.layers import conv2d_transpose, conv2d

nlstm = 1024


class EnvNetwork:
    def __init__(self, session, action_space_size, nbatch, K, nsteps, reuse):
        self.input_frames = tf.placeholder(tf.uint8, [nbatch, 84, 84, 9])
        self.input_actions = [tf.placeholder(tf.int32, [nbatch]) for _ in range(K)]
        self.real_frame = [tf.placeholder(tf.uint8, [nbatch, 84, 84, 1]) for _ in range(K)]
        self.masks = [tf.placeholder(tf.float32, [nbatch]) for _ in range(K)]
        self.hidden_states = tf.placeholder(tf.float32, [nbatch // nsteps, nlstm*2])
        self.lr = tf.placeholder(tf.float32, shape=[])

        #self.normed_input_frame = tf.squeeze(tf.one_hot(self.input_frames // norm_factor, 9))
        self.normed_input_frame = tf.cast(self.input_frames, tf.float32)
        self.normed_real_frame = [tf.squeeze(tf.one_hot(self.real_frame[i] // norm_factor, 9)) for i in range(K)]

        def forward(in_fr, in_ac, in_msk, hid, reuse):
            activ = tf.nn.relu
            encoding_1 = activ(conv(in_fr, 'encoding1', nf=32, rf=8, stride=2, pad='SAME', reuse=reuse))#42
            encoding_2 = activ(conv(encoding_1, 'encoding2', nf=64, rf=6, stride=2, pad='SAME', reuse=reuse)) #21
            encoding_3 = activ(conv(encoding_2, 'encoding3', nf=64, rf=4, stride=2, pad='SAME', reuse=reuse)) #nbatch x 11 x 11 x 64

            xs = tf.reshape(encoding_3, [nbatch, -1])
            xs = fc(xs, 'encoding4', nlstm, reuse=reuse) #nbatch x 1024
            xs = batch_to_seq(xs, nbatch // nsteps, nsteps)
            ms = batch_to_seq(in_msk, nbatch // nsteps, nsteps)

            h5, hidden_states_new = lstm(xs, ms, hid, 'lstm', nh=nlstm, reuse=reuse)
            encoded_frame = seq_to_batch(h5)

            encoded_action = fc(tf.one_hot(in_ac, depth=action_space_size), 'action1', nlstm, reuse=reuse)
            encoded = tf.multiply(encoded_frame, encoded_action)
            decoding_0 = fc(encoded, 'decoding0', 11*11*64, reuse=reuse)
            decoding_0 = tf.reshape(decoding_0, [nbatch, 11, 11, 64])
            with tf.variable_scope('', reuse=reuse):
                decoding1_kernel = tf.get_variable("decode1", [4, 4, 64, 64], initializer=ortho_init(1.0))
                decoding1_bias = tf.get_variable("decode1b", [1, 1, 1, 64], initializer=tf.constant_initializer(0.0))
                decoding_1 = activ(decoding1_bias + tf.nn.conv2d_transpose(decoding_0, decoding1_kernel, output_shape=tf.shape(encoding_2), strides=[1,2,2,1]))
                #decoding_1 = tf.nn.conv2d_transpose(decoding_0, decoding1_kernel, output_shape=tf.shape(encoding_2), strides=[1,2,2,1])
                decoding2_kernel = tf.get_variable("decode2", [6, 6, 32, 64], initializer=ortho_init(1.0))
                decoding2_bias = tf.get_variable("decode2b", [1, 1, 1, 32], initializer=tf.constant_initializer(0.0))
                decoding_2 = activ(decoding2_bias + tf.nn.conv2d_transpose(decoding_1, decoding2_kernel, output_shape=tf.shape(encoding_1), strides=[1,2,2,1]))
                #decoding_2 = tf.nn.conv2d_transpose(decoding_1, decoding2_kernel, output_shape=tf.shape(encoding_1), strides=[1,2,2,1])
                decoding3_kernel = tf.get_variable("decode3", [8, 8, 9, 32], initializer=ortho_init(1.0))
                decoding3_bias = tf.get_variable("decode3b", [1, 1, 1, 9], initializer=tf.constant_initializer(0.0))
                decoding_3 = activ(decoding3_bias + tf.nn.conv2d_transpose(decoding_2, decoding3_kernel, output_shape=tf.shape(self.normed_real_frame[0]), strides=[1,2,2,1]))
                #decoding_3 = tf.nn.conv2d_transpose(decoding_2, decoding3_kernel, output_shape=tf.shape(self.normed_real_frame[0]), strides=[1,2,2,1])

            return decoding_3, hidden_states_new

        def forward_multistep(in_fr, in_ac, in_msk, hid):
            all_outputs = []
            for i in range(K):
                if i == 0:
                    onehot_out, hidden_state = forward(in_fr, in_ac[i], in_msk[i], hid, reuse=reuse)
                    out_hidden_state = hidden_state
                else:
                    onehot_out, hidden_state = forward(onehot_out, in_ac[i], in_msk[i], hidden_state, reuse=True)

                all_outputs.append(onehot_out)
            return all_outputs, out_hidden_state

        all_outputs, new_hidden_state = forward_multistep(self.normed_input_frame, self.input_actions, self.masks, self.hidden_states)

        all_loss = 0
        for i in range(K):
            all_loss += tf.nn.softmax_cross_entropy_with_logits(labels=self.normed_real_frame[i], logits=all_outputs[i])
        with tf.variable_scope('', reuse=reuse):
            train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(all_loss)

        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.sess = session

        def train_and_predict(input_frames, input_actions, masks, expected_obs, hidden_states):
            # prediction: list len=K
            base_dict = {}
            for i in range(K):
                base_dict[self.real_frame[i]] = expected_obs[i]
                base_dict[self.input_actions[i]] = input_actions[i]
                base_dict[self.masks[i]] = masks[i]

            base_dict.update({self.input_frames: input_frames, self.hidden_states: hidden_states, self.lr: 1e-4})
            _, prediction_loss, prediction, hs = self.sess.run([train_op, all_loss, all_outputs, new_hidden_state], feed_dict=base_dict)
            return np.asarray(prediction_loss).sum(), prediction, hs

        def predict(input_frames, input_actions, masks, hidden_states):
            base_dict = {}
            for i in range(K):
                base_dict[self.input_actions[i]] = input_actions[i]
                base_dict[self.masks[i]] = masks[i]

            base_dict.update({self.input_frames: input_frames, self.hidden_states: hidden_states})
            prediction, hs = self.sess.run([all_outputs, new_hidden_state], feed_dict=base_dict)
            return prediction, hs

        self.train_and_predict = train_and_predict
        self.predict = predict
        self.forward = forward
        self.initial_state = np.zeros((nbatch // nsteps, nlstm*2), dtype=np.float32)

    def _checkpoint_filename(self, episode):
        return 'O:/Doom/envmodel/flatv2/flat%s' % (episode)

    def save(self):
        variables = [v for v in tf.trainable_variables()]
        variable_value = self.sess.run(variables)
        joblib.dump(variable_value, 'O:\\Doom\\envmodel\\envmodel.dat')

    def restore(self):
        variables = [v for v in tf.trainable_variables()]
        variable_value = joblib.load('O:\\Doom\\envmodel\\envmodel.dat')
        restores = []
        for p, loaded_p in zip(variables, variable_value):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)
