from model.envNetwork import EnvNetwork
from baselines.ppo2.policies import *
import tensorflow as tf
from model.util import norm_factor
from gym.spaces.box import Box

K = 5
nlstm = 1024

class I2ANetwork:
    def __init__(self, sess, observation_space, action_space, nbatch, nsteps, reuse, model_name):
        envModel = EnvNetwork(sess, action_space_size=action_space.n, nbatch=nbatch, K = 1, nsteps=nsteps, reuse=reuse)
        envModel.restore()
        def rolloutpolicy(X, reuse):
            with tf.variable_scope('rolloutpolicy', reuse=reuse):
                h = nature_cnn(X)
                pi = fc(h, 'pi', action_space.n, init_scale=0.01)
                return pi

        input_frame = tf.placeholder(tf.uint8, [nbatch, 84, 84, 4])
        normed_input_frame = tf.cast(tf.squeeze(tf.one_hot(input_frame[:,:,:,-1:] // norm_factor, 9)), tf.float32)
        hidden_state = tf.placeholder(tf.float32, [nbatch//nsteps , nlstm*2])

        #only for env model
        input_action = tf.placeholder(tf.uint8, [nbatch])
        masks = tf.placeholder(tf.float32, [nbatch])

        #calculate rollout actions
        dummy_mask = tf.constant([0 for _ in range(nbatch)], dtype=tf.float32)

        predicted_frames = [[] for _ in range(action_space.n)]
        for action in range(action_space.n):
            for steps in range(K):
                if steps == 0:
                    output_frame, output_hidden = envModel.forward(normed_input_frame, tf.constant([action for _ in range(nbatch)]),
                                                                   dummy_mask, hidden_state, reuse=True)
                else:
                    output_frame, output_hidden = envModel.forward(output_frame, rollout_action, dummy_mask, output_hidden, reuse=True)
                predicted_frames[action].append(tf.stop_gradient(output_frame))
                with tf.variable_scope(model_name):
                    if action == 0 and steps == 0:
                        pi = tf.stop_gradient(rolloutpolicy(output_frame, reuse = reuse))
                    else:
                        pi = tf.stop_gradient(rolloutpolicy(output_frame, reuse = True))
                pdtype = make_pdtype(action_space)
                pd = pdtype.pdfromflat(pi)
                rollout_action = pd.sample()

        _, new_hidden_state = envModel.forward(normed_input_frame, input_action, masks, hidden_state, reuse=True)
        with tf.variable_scope(model_name, reuse=reuse):
            def encode(rollout, reuse):
                all_seq = []
                initial_hidden = tf.constant(np.zeros([nbatch, 512*2]), dtype=tf.float32)
                cnt = 0
                for prediction in rollout:
                    activ = tf.nn.relu
                    if cnt > 0:
                        int_reuse = True
                    else:
                        int_reuse = reuse
                    encoding_1 = activ(
                        conv(prediction, 'encode1', nf=32, rf=8, stride=2, pad='SAME', reuse=int_reuse))  # 42
                    encoding_2 = activ(
                        conv(encoding_1, 'encode2', nf=64, rf=6, stride=2, pad='SAME', reuse=int_reuse))  # 21
                    encoding_3 = activ(conv(encoding_2, 'encode3', nf=64, rf=4, stride=2, pad='SAME',
                                            reuse=int_reuse))  # nbatch x 11 x 11 x 64
                    xs = tf.reshape(encoding_3, [nbatch, -1])
                    xs = fc(xs, 'encode4', 512, reuse=int_reuse)  # nbatch x 512
                    xs = batch_to_seq(xs, nbatch, 1)
                    all_seq.extend(xs)
                    cnt += 1

                assert (cnt == K)
                encoding_dummy_mask = tf.constant([0 for _ in range(nbatch*K)], dtype=tf.float32)
                ms = batch_to_seq(encoding_dummy_mask, nbatch, K)
                h5, _ = lstm(all_seq, ms, initial_hidden, 'encode_lstm', nh = 512, reuse=reuse)
                return h5[-1]

            #model based path
            encoded = [] #element: batch_size x nhidden
            for action in range(action_space.n):
                if action == 0:
                    encoded.append(encode(reversed(predicted_frames[action]), reuse=reuse))
                else:
                    encoded.append(encode(reversed(predicted_frames[action]), reuse=True))
            concat_encoded = tf.concat(encoded, axis=-1) #batch_size x (5*nhidden)

            #model free path
            with tf.variable_scope('modelfree'):
                h = nature_cnn(input_frame)
                total_h = tf.concat([concat_encoded, h], axis=-1)
                pi = fc(total_h, 'pi', action_space.n, init_scale=0.01)
                vf = fc(total_h, 'v', 1)[:,0]

            pdtype = make_pdtype(action_space)
            pd = pdtype.pdfromflat(pi)
            a0 = pd.sample()

            #training rollout policy
            rollout_ac = rolloutpolicy(normed_input_frame, reuse=True)
            rollout_pd = pdtype.pdfromflat(rollout_ac)

            #interface
            self.train_rolloutpolicy = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(rollout_pd.neglogp(a0))
            self.X = input_frame
            self.pi = pi
            self.vf = vf
            self.S = hidden_state
            self.M = masks
            self.initial_state = np.zeros([nbatch//nsteps, 2*nlstm], np.float32)
            def step(ob, hid, msk):
                a, v = sess.run([a0, vf], {input_frame:ob, hidden_state:hid})
                nh = sess.run(new_hidden_state, {input_frame:ob, hidden_state:hid, masks:msk, input_action:a})
                return a, v, nh, None

            def value(ob, hid, msk):
                return sess.run(vf, {input_frame:ob, hidden_state:hid})

            self.step = step
            self.value = value