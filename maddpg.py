from make_env import make_env
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import threading
import os
from tqdm import tqdm
from time import sleep

from utils import save, load, AgentBase


""" =============================== GLOBAL VARIABLES =========================== """
N_AGENTS = 3
GAMMA = 0.9
TAU = 0.01
VAR_DECAY = 0.995
VAR_INIT = 1.0
A_LR = 1e-4
C_LR = 2e-4
A_ITER = 1
C_ITER = 5
A_DIM = 5
S_DIM = 16

BATCH_SIZE = 128
EP_MAXLEN = 300
N_ITERS = 200000
CAPACITY = 50000
WRITE_LOGS_EVERY = 200
LOGDIR = './logs/maddpg/'
MODEL_DIR = './ckpt/maddpg/'

RENDER = True

""" ============================================================================= """

class EMAGetter(object):
    ema = tf.train.ExponentialMovingAverage(decay=1.0 - TAU)

    def __call__(self, getter, name, *args, **kwargs):
        return self.ema.average(getter(name, *args, **kwargs))


class MemoryBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.state = [np.zeros((capacity, S_DIM), dtype=np.float32) for _ in range(N_AGENTS)]
        self.action = np.zeros((capacity, N_AGENTS * A_DIM), dtype=np.float32)
        self.reward = [np.zeros((capacity, 1), dtype=np.float32) for _ in range(N_AGENTS)]
        self.s_next = [np.zeros((capacity, S_DIM), dtype=np.float32) for _ in range(N_AGENTS)]

        self.mutex = threading.Lock()
        self.pointer = 0

    def init_buffer(self, env, model):
        print(' [*] Start to initialize memory buffer...')
        with tqdm(total=self.capacity) as pbar:
            while True:
                s = env.reset()
                for it in range(EP_MAXLEN):
                    a = model.choose_action(s)
                    s_next, r, done, info = env.step(a)
                    self.store_transition(s, a, r, s_next)
                    pbar.update(1)
                    if self.pointer == 0:
                        return
                    if done:
                        break

    def store_transition(self, ss, aa, rr, ss_next):
        with self.mutex:
            self.action[self.pointer, :] = aa[: -1, :].flatten()
            for s, r, s_next, state, reward, next in zip(ss, rr, ss_next, self.state, self.reward, self.s_next):
                state[self.pointer, :] = s
                reward[self.pointer, :] = r
                next[self.pointer, :] = s_next
            self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, num):
        indices = np.random.randint(0, self.capacity, size=[num])
        ss, rr, ss_next = [], [], []
        for it in range(N_AGENTS):
            ss.append(self.state[it][indices])
            rr.append(self.reward[it][indices])
            ss_next.append(self.s_next[it][indices])
        return ss, self.action[indices], rr, ss_next



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
        self.name = name


    def add_critic(self, actions):
        with tf.variable_scope(self.name):
            self.target_q = self._build_q_network(self.state, actions, 'Critic')

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.name + '/Actor')
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/Critic')
        self.target_update = [self.ema_getter.ema.apply(a_vars), self.ema_getter.ema.apply(c_vars)]

        with tf.variable_scope(self.name):
            self.a_next = self._build_policy(self.s_next, 'Actor', A_DIM, trainable=False,
                                             reuse=True, custom_getter=self.ema_getter)


    def add_control(self, actions):
        with tf.variable_scope(self.name):
            self.eval_q = self._build_q_network(self.s_next, actions, 'Critic', trainable=False,
                                                reuse=True, custom_getter=self.ema_getter)

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.name + '/Actor')
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/Critic')
        self.a_loss = -tf.reduce_mean(self.target_q)
        self.a_optim = tf.train.AdamOptimizer(A_LR).minimize(self.a_loss, var_list=a_vars)

        with tf.control_dependencies(self.target_update):
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


    def _build_q_network(self, state, action, scope, trainable=True, reuse=False, custom_getter=None):
        with tf.variable_scope(scope, reuse=reuse, custom_getter=custom_getter):
            s_hidden = tf.layers.dense(state, 64, activation=None, trainable=trainable, name='s_hidden')
            a_hidden = tf.layers.dense(action, 64, activation=None, use_bias=False,
                                       trainable=trainable, name='a_hidden')
            h2 = tf.nn.relu(s_hidden + a_hidden)
            h3 = tf.layers.dense(h2, 64, activation=tf.nn.relu, trainable=trainable, name='h2')
            return tf.layers.dense(h3, 1, trainable=trainable, use_bias=True, activation=None, name='h3')



class MADDPGModel(AgentBase):
    name = 'MADDPGModel'

    def __init__(self):
        self.agents = [Agent('Predator_{}'.format(i)) for i in range(N_AGENTS)]
        self.actors = tf.concat([agent.actor for agent in self.agents], axis=-1)
        for agent in self.agents:
            agent.add_critic(self.actors)
        self.actors_next = tf.concat([agent.a_next for agent in self.agents], axis=-1)
        for agent in self.agents:
            agent.add_control(self.actors_next)

        self.a_optim = [agent.a_optim for agent in self.agents]
        self.c_optim = [agent.c_optim for agent in self.agents]
        self.counter = 0
        self.variance = VAR_INIT

        mean_rewards = [tf.reduce_mean(agent.reward) for agent in self.agents]
        r_sums = [tf.summary.scalar('reward_%d' % i, r) for i, r in enumerate(mean_rewards)]
        q_sums = [tf.summary.scalar('critic_loss_%d' % i, agent.td_error) for i, agent in enumerate(self.agents)]
        a_sums = [tf.summary.scalar('actor_loss_%d' % i, agent.a_loss) for i, agent in enumerate(self.agents)]
        self.sums = tf.summary.merge(r_sums + q_sums + a_sums, name='summaries')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Build DDPGModel finished...')


    def train(self, ss, aa, rr, ss_next, writer=None):
        feed_dict = dict()
        for agent, s, r, s_next in zip(self.agents, ss, rr, ss_next):
            feed_dict[agent.state] = s
            feed_dict[agent.reward] = r
            feed_dict[agent.s_next] = s_next
        for _ in range(A_ITER):
            self.sess.run(self.a_optim, feed_dict=feed_dict)
        feed_dict[self.actors] = aa
        for _ in range(C_ITER):
            self.sess.run(self.c_optim, feed_dict=feed_dict)
        self.counter += 1
        if writer is not None and self.counter % WRITE_LOGS_EVERY == (WRITE_LOGS_EVERY - 1):
            self.variance *= VAR_DECAY
            sumstr = tf.Summary()
            sumstr.value.add(simple_value=self.variance, tag='MADDPG/exploration_noise')
            writer.add_summary(sumstr, global_step=self.counter)
            sumstr = self.sess.run(self.sums, feed_dict=feed_dict)
            writer.add_summary(sumstr, global_step=self.counter)


    def choose_action(self, states):
        a_res = np.zeros((N_AGENTS + 1, A_DIM), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            a = self.sess.run(agent.actor, feed_dict={agent.state: states[i][None, :]})
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
    buffer = MemoryBuffer(CAPACITY)
    buffer.init_buffer(env, model)
    coord = tf.train.Coordinator()
    _, model.counter = load(model.sess, MODEL_DIR)
    slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)

    class ModelThread(threading.Thread):
        def __init__(self, wid=0):
            self.wid = wid
            super(ModelThread, self).__init__()
            self.writer = tf.summary.FileWriter(LOGDIR, model.sess.graph)
            print(' [*] ModelThread wid {} okay...'.format(wid))

        def run(self):
            print(' [*] ModelThread start to run...')
            for it in range(N_ITERS):
                ss, aa, rr, ss_next = buffer.sample(BATCH_SIZE)
                model.train(ss, aa, rr, ss_next, writer=self.writer)
            coord.request_stop()
            print(' [*] ModelThread wid {} reaches the exit!'.format(self.wid))


    class BufferThread(threading.Thread):
        def __init__(self, wid=1):
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
                    a = model.choose_action(s)
                    s_next, r, done, info = self.env.step(a)
                    buffer.store_transition(s, a, r, s_next)
                    s = s_next
                    if done:
                        break
            print(' [*] BufferThread wid {} reaches the exit!'.format(self.wid))


    model_thread = ModelThread()
    buffer_thread = BufferThread()
    model_thread.start()
    buffer_thread.start()
    coord.join([model_thread, buffer_thread])

    save(model.sess, MODEL_DIR, model.name, global_step=model.counter)

    while True:
        # Need to interrupt mannually
        s = env.reset()
        for it in range(EP_MAXLEN):
            env.render()
            s, r, info, done = env.step(model.choose_action(s))
            sleep(0.2)
            if any(done):
                break