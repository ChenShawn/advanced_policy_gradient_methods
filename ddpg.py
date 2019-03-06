import tensorflow as tf
import gym
from tensorflow.contrib import slim
from gym.wrappers import Monitor
import argparse
import numpy as np
import os
import random
import threading

from utils import save, load, set_global_seed
from evaluate import AgentEvaluator

""" =============== GLOBAL VARIABLES ================= """
GAMMA = 0.9
A_LR = 1e-4
C_LR = 1e-4
A_ITER = 5
C_ITER = 5
A_DIM = 1
S_DIM = 3

BATCH_SIZE = 32
EPSILON = 0.2
EP_MAXLEN = 200
N_ITERS = 300000
CAPACITY = 10000
WRITE_LOGS_EVERY = 100
LOGDIR = './logs/ddpg/'
MODEL_DIR = './ckpt/ddpg/'

RENDER = True
LOCK = threading.Lock()

""" ================================================== """

class MemoryBuffer(object):
    def __init__(self, capacity, s_dim):
        self.capacity = capacity
        self.state = np.zeros((capacity, s_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.s_next = np.zeros((capacity, s_dim), dtype=np.float32)
        self.pointer = 0

    def init_buffer(self, env, agent):
        while True:
            s = env.reset()
            discounted_r = agent.get_value(s)
            gamma = 1.0
            for it in range(EP_MAXLEN):
                s_next, r, done, info = env.step(agent.choose_action(s))
                r = (r + 8.0) / 8.0
                discounted_r += gamma * r
                gamma *= GAMMA
                self.store_transition(s, discounted_r, s_next)
                if self.pointer == 0:
                    return
                if done:
                    break

    def store_transition(self, s, r, s_next):
        self.state[self.pointer, :] = s
        self.reward[self.pointer, :] = r
        self.s_next[self.pointer, :] = s_next
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, num):
        indices = np.random.randint(0, self.capacity, size=[num])
        return self.state[indices], self.reward[indices], self.s_next[indices]


class Actor(object):
    def __init__(self, state, s_next, a_dim):
        with tf.variable_scope('Actor'):
            self.action = self._build_policy(state, 'target_pi', a_dim)
            self.action_old = self._build_policy(state, 'eval_pi', a_dim, trainable=False)
            self.a_next = self._build_policy(s_next, 'target_pi', a_dim, trainable=False, reuse=True)

        train_vars = tf.trainable_variables(scope='Actor')
        self.pi_vars = [var for var in train_vars if 'target_pi' in var.name]
        self.fixed_vars = [var for var in train_vars if 'eval_pi' in var.name]
        self.assign_op = [fixed.assign(pi) for fixed, pi in zip(self.fixed_vars, self.pi_vars)]
        print(' [*] Build actor finished...')


    def _build_policy(self, state, scope, a_dim, trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = tf.layers.dense(state, 64, activation=tf.nn.relu, trainable=trainable,
                                  kernel_initializer=tf.orthogonal_initializer(),
                                  bias_initializer=tf.constant_initializer(0.001))
            action = 2.0 * tf.layers.dense(net, a_dim, activation=tf.nn.tanh, trainable=trainable,
                                           kernel_initializer=tf.orthogonal_initializer())
        return action


    def add_optim(self, critic):
        self.loss = -tf.reduce_mean(critic)
        self.optim = tf.train.AdamOptimizer(A_LR).minimize(self.loss, var_list=self.pi_vars)
        print(' [*] Actor optimizer initialization finished...')


class Critic(object):
    def __init__(self, state, action, reward, s_next, a_next):
        with tf.variable_scope('Critic'):
            self.target_q = self._build_q_network(s_next, a_next, 'target_q')
            self.eval_q = self._build_q_network(state, action, 'eval_q', trainable=False)

        train_vars = tf.trainable_variables(scope='Critic')
        self.q_vars = [var for var in train_vars if 'target_q' in var.name]
        self.fixed_vars = [var for var in train_vars if 'eval_q' in var.name]

        self.assign_op = [fixed.assign(q) for fixed, q in zip(self.fixed_vars, self.q_vars)]
        self.critic = reward + GAMMA * self.target_q - self.eval_q
        self.loss = tf.reduce_mean(tf.squared_difference(reward + GAMMA * self.target_q, self.eval_q))
        self.optim = tf.train.AdamOptimizer(C_LR).minimize(self.loss, var_list=self.q_vars)
        print(' [*] Build critic finished...')


    def _build_q_network(self, state, action, scope, trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            h1 = tf.layers.dense(state, 64, activation=None, trainable=trainable,
                                 kernel_initializer=tf.orthogonal_initializer())
            h2 = tf.layers.dense(action, 64, activation=None, use_bias=False, trainable=trainable,
                                 kernel_initializer=tf.orthogonal_initializer())
            h3 = tf.nn.relu(h1 + h2)
            return tf.layers.dense(h3, 1, trainable=trainable, use_bias=True, activation=None,
                                   kernel_initializer=tf.orthogonal_initializer())



class DDPGModel(object):
    name = 'DDPGModel'

    def __init__(self, s_dim, a_dim):
        self.state = tf.placeholder(tf.float32, [None, s_dim], name='state')
        self.s_next = tf.placeholder(tf.float32, [None, s_dim], name='s_next')
        self.reward = tf.placeholder(tf.float32, [None, 1], name='discounted_r')

        self.actor = Actor(self.state, self.s_next, a_dim)
        self.critic = Critic(self.state, self.actor.action, self.reward, self.s_next, self.actor.a_next)
        self.actor.add_optim(self.critic.critic)

        self.sums = tf.summary.merge([
            tf.summary.scalar('reward', tf.reduce_mean(self.reward)),
            tf.summary.scalar('critic', tf.reduce_mean(self.critic.critic)),
            tf.summary.histogram('critic', self.critic.critic),
            tf.summary.scalar('actor_loss', self.actor.loss),
            tf.summary.scalar('critic_loss', self.critic.loss),
            tf.summary.histogram('Q_taregt', self.critic.target_q)
        ], name='summaries')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Build DDPGModel finished...')


    def train(self, s, r, s_next, callback=None):
        feed_dict = {self.state: s, self.reward: r, self.s_next: s_next}
        for _ in range(C_ITER):
            self.sess.run(self.critic.optim, feed_dict=feed_dict)
        self.sess.run(self.critic.assign_op)
        for _ in range(A_ITER):
            self.sess.run(self.actor.optim, feed_dict=feed_dict)
        self.sess.run(self.actor.assign_op)

        if callback is not None:
            callback(self.sums, feed_dict)


    def choose_action(self, s):
        """choose_action
        Try epsilon-greedy for more exploration
        """
        if random.uniform(0.0, 1.0) < EPSILON:
            return np.array([random.uniform(0.0, 1.0)], dtype=np.float32)
        else:
            return self.sess.run(self.actor.action, feed_dict={self.state: s[None, :]})[0]


    def get_value(self, s):
        return self.sess.run(self.critic.target_q, feed_dict={self.state: s[None, :]})



class CallbackFunctor(object):
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir, model.sess.graph)
        self.counter = 0

    def __call__(self, sums, feed_dict):
        if self.counter % WRITE_LOGS_EVERY == 5:
            sumstr = model.sess.run(sums, feed_dict=feed_dict)
            self.writer.add_summary(sumstr, global_step=self.counter)
        self.counter += 1



if __name__ == '__main__':
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model = DDPGModel(S_DIM, A_DIM)
    buffer = MemoryBuffer(CAPACITY, S_DIM)
    coord = tf.train.Coordinator()
    load(model.sess, model_path=MODEL_DIR)
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
                s, r, s_next = buffer.sample(BATCH_SIZE)
                LOCK.release()
                model.train(s, r, s_next, callback=self.functor)
            coord.request_stop()
            print(' [*] ModelThread wid {} reaches the exit!'.format(self.wid))


    class BufferThread(threading.Thread):
        def __init__(self, wid=1, render=True):
            self.render = render
            self.wid = wid
            self.env = gym.make('Pendulum-v0').unwrapped
            self.env.seed(1)
            super(BufferThread, self).__init__()
            print(' [*] BufferThread {} okay...'.format(wid))

        def run(self):
            print(' [*] BufferThread start to run...')
            while not coord.should_stop():
                s = self.env.reset()
                gamma = 1.0
                discounted_r = model.get_value(s)
                for it in range(EP_MAXLEN):
                    if self.render:
                        self.env.render()
                    s_next, r, done, info = self.env.step(model.choose_action(s))
                    r = (r + 8.0) / 8.0
                    discounted_r += gamma * r
                    gamma *= GAMMA
                    LOCK.acquire()
                    buffer.store_transition(s, discounted_r, s_next)
                    LOCK.release()
                    s = s_next
                    if done:
                        break
            print(' [*] BufferThread wid {} reaches the exit!'.format(self.wid))


    model_thread = ModelThread()
    buffer_thread = BufferThread(render=RENDER)
    buffer.init_buffer(buffer_thread.env, model)

    model_thread.start()
    buffer_thread.start()
    coord.join([model_thread, buffer_thread])
    save(model.sess, MODEL_DIR, model.name, global_step=N_ITERS)
    print(' [*] The main process reaches the exit!!')