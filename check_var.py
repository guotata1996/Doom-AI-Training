import tensorflow as tf
from model.i2aNetwork import I2ANetwork
from gym.spaces.discrete import Discrete

sess = tf.Session()
i2a = I2ANetwork(sess, None, Discrete(6), 72,12,False,'prediction_flat')
sess.run(tf.global_variables_initializer())
variable_name = [c.name for c in tf.trainable_variables()]
for name in variable_name:
    if name.startswith('prediction_flat'):
        print(name)