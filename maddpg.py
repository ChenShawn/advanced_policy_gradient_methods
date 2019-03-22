from make_env import make_env
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import threading
import os

from utils import save, load, AgentBase


""" =============================== GLOBAL VARIABLES =========================== """
N_AGENTS = 3
GAMMA = 0.9
TAU = 0.01
VAR_DECAY = 0.995
A_LR = 1e-3
C_LR = 2e-3
A_ITER = 3
C_ITER = 1
A_DIM = 5
S_DIM = 16

BATCH_SIZE = 128
EP_MAXLEN = 300
N_ITERS = 9000
CAPACITY = 100000
WRITE_LOGS_EVERY = 200
LOGDIR = './logs/ddpg/'
MODEL_DIR = './ckpt/ddpg/'

RENDER = True
TEST = True

""" ============================================================================= """

class EMAGetter(object):
    ema = tf.train.ExponentialMovingAverage(decay=1.0 - TAU)

    def __call__(self, getter, name, *args, **kwargs):
        return self.ema.average(getter(name, *args, **kwargs))


class MemoryBuffer(object):
    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity
        self.state = np.zeros((capacity, s_dim), dtype=np.float32)
        self.action = np.zeros((capacity, a_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.s_next = np.zeros((capacity, s_dim), dtype=np.float32)

        self.mutex = threading.Lock()
        self.pointer = 0

    def init_buffer(self, env, model):
        while True:
            s = env.reset()
            for it in range(EP_MAXLEN):
                a = model.choose_action(s)
                s_next, r, done, info = env.step(a)
                r = (r + 8.0) / 8.0
                self.store_transition(s, a, r, s_next)
                if self.pointer == 0:
                    return
                if done:
                    break

    def store_transition(self, s, a, r, s_next):
        with self.mutex:
            self.state[self.pointer, :] = s
            self.action[self.pointer, :] = a
            self.reward[self.pointer, :] = r
            self.s_next[self.pointer, :] = s_next
            self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, num):
        with self.mutex:
            indices = np.random.randint(0, self.capacity, size=[num])
            return self.state[indices], self.action[indices], self.reward[indices], self.s_next[indices]



class Agent(AgentBase):
    """Agent
    Definition for a single agent
    """
    def __init__(self, name):
        self.ema_getter = EMAGetter()
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, S_DIM], name='state')
            self.s_next = tf.placeholder(tf.float32, [None, S_DIM], name='s_next')
            self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
            self.actor = self._build_policy(self.state, 'Actor', A_DIM)
        self.buffer = MemoryBuffer(CAPACITY, S_DIM, A_DIM)
        self.name = name


    def add_critic(self, actors):
        with tf.variable_scope(self.name):
            self.target_q = self._build_q_network(self.state, actors, 'Critic')

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.name + '/Actor')
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/Critic')
        target_update = [self.ema_getter.ema.apply(a_vars), self.ema_getter.ema.apply(c_vars)]

        with tf.variable_scope(self.name):
            self.a_next = self._build_policy(self.s_next, 'Actor', A_DIM, trainable=False,
                                             reuse=True, custom_getter=self.ema_getter)
            self.eval_q = self._build_q_network(self.s_next, self.a_next, 'Critic', trainable=False,
                                                reuse=True, custom_getter=self.ema_getter)

        a_loss = -tf.reduce_mean(self.target_q)
        self.a_optim = tf.train.AdamOptimizer(A_LR).minimize(a_loss, var_list=a_vars)

        with tf.control_dependencies(target_update):
            self.td_error = tf.losses.mean_squared_error(labels=self.reward + GAMMA * self.eval_q,
                                                         predictions=self.target_q)
            self.c_optim = tf.train.AdamOptimizer(C_LR).minimize(self.td_error, var_list=c_vars)
        print(' [*] Agent {} built finished...'.format(self.name))


    def _build_policy(self, state, scope, a_dim, trainable=True, reuse=False, custom_getter=None):
        with tf.variable_scope(scope, reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(state, 64, activation=tf.nn.relu, trainable=trainable, name='h1')
            net = tf.layers.dense(net, 64, activation=tf.nn.relu, trainable=trainable, name='h2')
            action = tf.layers.dense(net, a_dim, activation=tf.nn.tanh, trainable=trainable, name='h3')
        return action


    def _build_q_network(self, state, action_list, scope, trainable=True, reuse=False, custom_getter=None):
        with tf.variable_scope(scope, reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(state, 64, activation=None, trainable=trainable, name='s_hidden')
            hiddens = [h1]
            for idx, action in enumerate(action_list):
                a_hidden = tf.layers.dense(action, 64, activation=None, use_bias=False,
                                           trainable=trainable, name='a_hidden_' + str(idx))
                hiddens.append(a_hidden)
            h2 = tf.nn.relu(tf.add_n(hiddens))
            h3 = tf.layers.dense(h2, 64, activation=tf.nn.relu, trainable=trainable, name='h2')
            return tf.layers.dense(h3, 1, trainable=trainable, use_bias=True, activation=None, name='h3')


    def train(self, sess, s, a, r, s_next):
        feed_dict = {self.state: s, self.actor: a, self.reward: r, self.s_next: s_next}
        for _ in range(C_ITER):
            sess.run(self.c_optim, feed_dict=feed_dict)
        feed_dict.pop(self.actor)
        for _ in range(A_ITER):
            sess.run(self.a_optim, feed_dict=feed_dict)



class MADDPGModel(AgentBase):
    name = 'DDPGModel'

    def __init__(self):
        self.agents = [Agent('Predator_{}'.format(i)) for i in range(N_AGENTS)]
        self.actors = [agent.actor for agent in self.agents]
        for agent in self.agents:
            agent.add_critic(self.actors)

        self.a_optim = [agent.a_optim for agent in self.agents]
        self.c_optim = [agent.c_optim for agent in self.agents]
        self.counter = 0
        self.variance = 1.0

        mean_rewards = [tf.reduce_mean(agent.reward) for agent in self.agents]
        r_sums = [tf.summary.scalar('reward_%d' % i, r) for i, r in enumerate(mean_rewards)]
        q_sums = [tf.summary.scalar('critic_loss_%d' % i, agent.td_error) for i, agent in enumerate(self.agents)]
        self.sums = tf.summary.merge(r_sums + q_sums, name='summaries')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Build DDPGModel finished...')


    def train(self, ss, aa, rr, ss_next, writer=None):
        for agent, s, a, r, s_next in zip(self.agents, ss, aa, rr, ss_next):
            agent.train(self.sess, s, a, r, s_next)
        self.counter += 1
        if writer is not None and self.counter % WRITE_LOGS_EVERY == (WRITE_LOGS_EVERY - 1):
            self.variance *= VAR_DECAY
            sumstr = tf.Summary().value.add(simple_value=self.variance, tag='MADDPG/exploration_noise')
            writer.add_summary(sumstr, global_step=self.counter)
            print('Global step {}, current variance in behavior policy {}'.format(self.counter, self.variance))
            feed_dict = {}
            sumstr = self.sess.run(self.sums, feed_dict=feed_dict)
            writer.add_summary(sumstr, global_step=self.counter)


    def choose_action(self, s):
        a_res = np.zeros((N_AGENTS, A_DIM), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            a = self.sess.run(agent.actor, feed_dict={agent.state: s[None, :]})
            a = np.clip(a + np.random.normal(0.0, self.variance, size=a.shape), -1.0, 1.0)
            a_res[i, 1] = a[0, 0]
            a_res[i, 3] = a[0, 1]
        return a_res



if __name__ == '__main__':
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    env = make_env('simple_tag').unwrapped
    model = MADDPGModel()
    coord = tf.train.Coordinator()
    _, model.counter = load(model.sess, MODEL_DIR)
    slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)

    class ModelThread(threading.Thread):
        def __init__(self, wid=0):
            self.wid = wid
            super(ModelThread, self).__init__()
            print(' [*] ModelThread wid {} okay...'.format(wid))

        def run(self):
            print(' [*] ModelThread start to run...')
            for it in range(N_ITERS):
                s, a, r, s_next = buffer.sample(BATCH_SIZE)
                model.train(s, a, r, s_next, callback=self.functor)
            coord.request_stop()
            print(' [*] ModelThread wid {} reaches the exit!'.format(self.wid))


    class BufferThread(threading.Thread):
        def __init__(self, wid=1, render=True):
            self.render = render
            self.wid = wid
            self.env = make_env('simple_tag').unwrapped
            self.env.seed(1)
            super(BufferThread, self).__init__()
            print(' [*] BufferThread {} okay...'.format(wid))

        def run(self):
            print(' [*] BufferThread start to run...')
            while not coord.should_stop():
                s = self.env.reset()
                for it in range(EP_MAXLEN):
                    if self.render:
                        self.env.render()
                    a = model.choose_action(s)
                    s_next, r, done, info = self.env.step(a)
                    r = (r + 8.0) / 8.0
                    buffer.store_transition(s, a, r, s_next)
                    s = s_next
                    if done:
                        break
            print(' [*] BufferThread wid {} reaches the exit!'.format(self.wid))