import os
import tensorflow as tf
import gym
from tensorflow.contrib import slim
from gym.wrappers import Monitor
import argparse
import numpy as np
import threading
import time

from utils import save, load, AgentBase

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" =========== DISTRIBUTED IMPLEMENTAION FOR SOFT Q-LEARNING ==============
=============================== GLOBAL VARIABLES =========================== """
GAMMA = 0.97
TAU = 0.01
LR = 2e-3
ALPHA = 1.0
S_SHAPE = [210, 160, 3]
A_DIM = 4
ENV_NAME = 'Breakout-v4'

BATCH_SIZE = 64
EP_MAXLEN = 2000
N_ITERS = 9000
CAPACITY = 3000
WRITE_LOGS_EVERY = 200
LOGDIR = './logs/sql/'
MODEL_DIR = './ckpt/sql/'

RENDER = False
TEST = True
LOCK = threading.Lock()

""" ========================================================================= """


class MemoryBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.state = np.zeros([capacity] + S_SHAPE, dtype=np.float32)
        self.action = np.zeros((capacity), dtype=np.int32)
        self.value_estimate = np.zeros((capacity), dtype=np.float32)
        self.pointer, self.n_sample, self.n_store = 0, 0, 0

    def store_transition(self, s, a, v_est):
        self.state[self.pointer] = s
        self.action[self.pointer] = a
        self.value_estimate[self.pointer] = v_est
        self.pointer = (self.pointer + 1) % self.capacity
        self.n_store += 1

    def sample(self, num):
        indices = np.random.randint(0, self.capacity, size=[num])
        self.n_sample += num
        return self.state[indices], self.action[indices], self.value_estimate[indices]



class SQLModel(AgentBase):
    name = 'SQLModel'

    def __init__(self):
        self.state = tf.placeholder(tf.float32, [None] + S_SHAPE, name='state')
        self.action = tf.placeholder(tf.int32, [None], name='action')
        self.s_next = tf.placeholder(tf.float32, [None] + S_SHAPE, name='s_next')
        self.value_estimate = tf.placeholder(tf.float32, [None], name='value_estimate')
        gray_state = self.image_process(self.state)
        gray_s_next = self.image_process(self.s_next)

        with tf.variable_scope('SQL'):
            self.target_q = self._build_q_network(gray_state, 'target_q')
            self.estimate_q = self._build_q_network(gray_s_next, 'estimate_q', trainable=False)
            pi_prob = tf.nn.softmax(self.estimate_q / ALPHA, axis=-1, name='soft_policy')
            self.policy = tf.distributions.Categorical(probs=pi_prob)
            self.execution = self.policy.sample(1)
            self.entropy = self.policy.entropy()

        tq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SQL/target_q')
        eq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SQL/estimate_q')
        self.q_update = [b.assign(a) for a, b in zip(tq_vars, eq_vars)]

        with tf.name_scope('soft_loss'):
            exp_logits = tf.exp(self.estimate_q / ALPHA)
            self.v_next = ALPHA * tf.log(tf.reduce_sum(exp_logits, axis=-1, name='value'))
            q_target = tf.reduce_sum(self.target_q * tf.one_hot(self.action, A_DIM), axis=-1)
            self.loss = tf.reduce_mean(tf.square(self.value_estimate - q_target))
            self.optim = tf.train.AdamOptimizer(LR).minimize(self.loss, var_list=tq_vars)
        self.counter = 0
        self.sums = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            tf.summary.scalar('train/entropy', tf.reduce_mean(self.entropy))
        ], name='summaries')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Build SQLModel finished...')


    def image_process(self, img_tensor, height=80, width=80):
        s_out = tf.image.rgb_to_grayscale(img_tensor)
        s_out = tf.image.crop_to_bounding_box(s_out, 34, 0, 160, 160)
        return tf.image.resize_images(s_out, (80, 80), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def _build_q_network(self, state, scope, trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            h1 = tf.layers.conv2d(state, 64, 3, padding='same', activation=tf.nn.relu, trainable=trainable, name='h1')
            h2 = tf.layers.conv2d(h1, 64, 3, padding='same', activation=tf.nn.relu, trainable=trainable, name='h2')
            h2_pool = tf.layers.max_pooling2d(h2, 2, 2, name='h2_pool')
            h3 = tf.layers.conv2d(h2_pool, 64, 3, padding='same', activation=tf.nn.relu, trainable=trainable, name='h3')
            h3_pool = tf.layers.max_pooling2d(h3, 2, 2, name='h3_pool')
            h4 = tf.layers.conv2d(h3_pool, 64, 3, padding='same', activation=tf.nn.relu, trainable=trainable, name='h4')
            h4_pool = tf.layers.max_pooling2d(h4, 2, 2, name='h4_pool')
            h5 = tf.layers.conv2d(h4_pool, 64, 3, padding='same', activation=tf.nn.relu, trainable=trainable, name='h5')
            h5_down = tf.layers.conv2d(h5, 1, 1, activation=tf.nn.relu, trainable=trainable, name='h5_down')
            h5_reshape = tf.reshape(h5_down, [-1, 100], name='h5_reshape')
            return tf.layers.dense(h5_reshape, A_DIM, activation=None, trainable=trainable, name='h6')

    def train(self, s, a, v, callback=None):
        feed_dict = {self.state: s, self.action: a, self.value_estimate: v}
        self.sess.run(self.optim, feed_dict=feed_dict)
        self.sess.run(self.optim, feed_dict=feed_dict)
        self.counter += 1
        if callback is not None:
            feed_dict[self.s_next] = s
            callback(self.sums, feed_dict)

    def choose_action(self, s):
        return self.sess.run(self.execution, feed_dict={self.s_next: s[None, :, :, :]})

    def get_value(self, s):
        return self.sess.run(self.v_next, feed_dict={self.s_next: s[None, :, :, :]})



class CallbackFunctor(object):
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir, model.sess.graph)
        self.counter = 0

    def __call__(self, sums, feed_dict):
        if self.counter % WRITE_LOGS_EVERY == 0:
            model.sess.run(model.q_update)
            sumstr = model.sess.run(sums, feed_dict=feed_dict)
            self.writer.add_summary(sumstr, global_step=self.counter)
        self.counter += 1



if __name__ == '__main__':
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model = SQLModel()
    buffer = MemoryBuffer(CAPACITY)
    coord = tf.train.Coordinator()
    _, model.counter = load(model.sess, model_path=MODEL_DIR)
    slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)

    class ModelThread(threading.Thread):
        def __init__(self, wid=0):
            self.wid = wid
            self.functor = CallbackFunctor(logdir=LOGDIR)
            super(ModelThread, self).__init__()
            print(' [*] ModelThread wid {} okay...'.format(wid))

        def run(self):
            print(' [*] ModelThread start to run...')
            for it in range(N_ITERS):
                LOCK.acquire()
                s, a, v = buffer.sample(BATCH_SIZE)
                LOCK.release()
                model.train(s, a, v, callback=self.functor)
            coord.request_stop()
            print(' [*] ModelThread wid {} reaches the exit!'.format(self.wid))


    class BufferThread(threading.Thread):
        def __init__(self, wid=1, render=True):
            self.render = render
            self.wid = wid
            self.env = gym.make(ENV_NAME).unwrapped
            #self.env.seed(1)
            super(BufferThread, self).__init__()
            print(' [*] BufferThread {} okay...'.format(wid))

        def run(self):
            print(' [*] BufferThread start to run...')
            while not coord.should_stop():
                s = self.env.reset()
                done = False
                s_list, a_list, r_list = [], [], []
                while not done:
                    if self.render:
                        self.env.render()
                    a = model.choose_action(s)
                    s_next, r, done, info = self.env.step(a)
                    s_list.append(s)
                    a_list.append(a)
                    r_list.append(r)
                    if r != 0:
                        print('BufferThread wid={} Back-tracking len={}...'.format(self.wid, len(r_list)))
                        total_reward = [model.get_value(s_next)[0]]
                        for t_r in r_list[::-1]:
                            total_reward.append(GAMMA * total_reward[-1] + t_r)
                        total_reward.pop(0)
                        total_reward = total_reward[::-1]
                        LOCK.acquire()
                        for t_s, t_a, t_r in zip(s_list, a_list, total_reward):
                            buffer.store_transition(t_s, t_a, t_r)
                        LOCK.release()
                        s_list.clear()
                        a_list.clear()
                        r_list.clear()
                    s = s_next
            print(' [*] BufferThread wid {} reaches the exit!'.format(self.wid))


    model_thread = ModelThread()
    buffer_thread = [BufferThread(wid=it, render=RENDER) for it in range(1, 2)]
    for thread in buffer_thread:
        thread.start()
    while not buffer.n_store >= buffer.capacity:
        time.sleep(10)
    buffer_thread[0].render = True

    model_thread.start()
    coord.join([model_thread] + buffer_thread)
    save(model.sess, MODEL_DIR, model.name, global_step=model.counter)
    print(' [*] The main process reaches the exit!!')
