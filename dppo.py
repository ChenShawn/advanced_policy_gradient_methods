import tensorflow as tf
import numpy as np
import gym
import argparse
import os
import multiprocessing
import queue
import signal
from datetime import datetime
import psutil

from utils import save, load


""" ================= GLOBAL VARIABLES ================== """
GLOBAL_QUEUE = queue.Queue()
UPDATE_EVENT = multiprocessing.Event()
ROLLING_EVENT = multiprocessing.Event()
QUEUE_EVENT = multiprocessing.Event()
TERM_EVENT = multiprocessing.Event()
N_ITER = 1000
GLOBAL_COUNTER = 0
PI_ITER = 10
V_ITER = 10
EP_MAXLEN = 200
MAX_QSIZE = 128
GAMMA = 0.9
BATCH_SIZE = 32
S_DIM, A_DIM = 3, 1

""" ===================================================== """


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--method', type=str, default='clip', help='either kl_pen or clip')
    parser.add_argument('--v_lr', type=float, default=2e-4, help='learning rate of value function update')
    parser.add_argument('--pi_lr', type=float, default=1e-4, help='learning rate of policy function')
    parser.add_argument('--gamma', type=float, default=0.9, help='discounted reward')
    parser.add_argument('--model_dir', type=str, default='./ckpt/dppo/', help='model directory')
    parser.add_argument('--logdir', type=str, default='./logs/dppo/', help='log directory')
    return parser.parse_args()


class PPOModel(multiprocessing.Process):
    def __init__(self, v_lr, pi_lr, s_dim, a_dim, method, logdir):
        super(PPOModel, self).__init__()
        self.state = tf.placeholder(tf.float32, [None, s_dim], 'state')
        self.action = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.method = method
        signal.signal(signal.SIGTERM, self.sigterm_callback)

        # Advantage function estimator
        with tf.variable_scope('value'):
            l1 = tf.layers.dense(self.state, 100, activation=tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            advantage = self.reward - self.v
            self.v_loss = tf.reduce_mean(tf.square(advantage))
            self.advantage = tf.stop_gradient(advantage)
            v_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')

        # Actor (Function pi(a|s) following behavior policy)
        pi, pi_vars = self._build_policy('pi', a_dim, trainable=True)
        oldpi, oldpi_vars = self._build_policy('oldpi', a_dim, trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_vars, oldpi_vars)]

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.action) / oldpi.prob(self.action)
                surr = ratio * self.advantage
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
            if method['name'] == 'kl_pen':
                self.lamda = tf.placeholder(tf.float32, None, 'lambda')
                self.pi_loss = -tf.reduce_mean(surr - self.lamda * kl)
            elif method['name'] == 'clip':
                self.pi_loss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.0 - method['epsilon'], 1.0 + method['epsilon']) * self.advantage
                ))
            else:
                raise NotImplementedError

        with tf.variable_scope('train_ops'):
            self.vtrain_op = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss, var_list=v_vars)
            self.pitrain_op = tf.train.AdamOptimizer(pi_lr).minimize(self.pi_loss, var_list=pi_vars)

        self.sums = tf.summary.merge([
            tf.summary.scalar('mean_reward', tf.reduce_mean(self.reward)),
            tf.summary.scalar('mean_advantage', tf.reduce_mean(self.advantage)),
            tf.summary.scalar('v_loss', self.v_loss),
            tf.summary.scalar('pi_loss', self.pi_loss),
            tf.summary.scalar('mean_kl', self.kl_mean),
            tf.summary.scalar('model_variance', tf.reduce_mean(pi._scale))
        ], name='summaries')

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Model built finished')
        self.writer = tf.summary.FileWriter(logdir, self.sess.graph)


    def run(self):
        global GLOBAL_COUNTER
        print(' [*] PPO model update process start to run...')
        while GLOBAL_COUNTER < N_ITER:
            # Block model update and release rolling event if global queue is empty
            if GLOBAL_QUEUE.empty():
                UPDATE_EVENT.clear()
                ROLLING_EVENT.set()
            UPDATE_EVENT.wait()

            while not GLOBAL_QUEUE.empty():
                self.sess.run(self.update_oldpi_op)
                s, a, r = GLOBAL_QUEUE.get()
                feed_dict = {self.state: s, self.action: a, self.reward: r}
                # Update policy using clipping surrogate
                for _ in range(PI_ITER):
                    self.sess.run(self.pitrain_op, feed_dict=feed_dict)
                # Update value network
                for _ in range(V_ITER):
                    self.sess.run(self.vtrain_op, feed_dict=feed_dict)
                # Collect summaries
                sumstr = self.sess.run(self.sums, feed_dict=feed_dict)
                self.writer.add_summary(sumstr, global_step=GLOBAL_COUNTER)
                GLOBAL_COUNTER += 1
            print(datetime.now(), '--Iteration {} --Global queue clear'.format(GLOBAL_COUNTER))
        TERM_EVENT.set()
        save(self.sess, './ckpt/dppo/', self.__name__, global_step=GLOBAL_COUNTER)


    def _build_policy(self, name, n_out, trainable=True, reuse=False):
        params = {'trainable': trainable, 'kernel_initializer': tf.orthogonal_initializer()}
        # params = {'trainable': trainable}
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(self.state, 100, tf.nn.relu, **params)
            mu = 2.0 * tf.layers.dense(l1, n_out, tf.nn.tanh, **params)
            sigma = tf.layers.dense(l1, n_out, tf.nn.softplus, **params)
            gaussian = tf.distributions.Normal(loc=mu, scale=sigma)
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return gaussian, vars


    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.sample_op, feed_dict={self.state: s})[0]


    def value_estimate(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, feed_dict={self.state: s})[0, 0]


    def sigterm_callback(self, signum, frame):
        """sigterm_callback
        Callback function
        Allow user to mannually terminate the process and save the model at any time
        Use `kill xxxx` instead of `kill -9 xxxx`
        """
        print(' [*] System signal {} caught, ready to save...'.format(signum))
        print(' [*] Frame: ', frame)
        save(self.sess, './ckpt/dppo/', 'Pendumlum-v0', global_step=GLOBAL_COUNTER)



class PPOWorker(multiprocessing.Process):
    def __init__(self, wid, model):
        super(PPOWorker, self).__init__()
        self.wid = wid
        self.env = gym.make('Pendulum-v0')
        self.model = model

    def run(self):
        print(' [*] Worker {} process start to run...'.format(self.wid))
        while not TERM_EVENT.is_set():
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            # Block data collection if PPOModel is updating
            if UPDATE_EVENT.is_set():
                ROLLING_EVENT.clear()
            ROLLING_EVENT.wait()

            for it in range(EP_MAXLEN):
                buffer_s.append(s[None, :])
                a = self.model.choose_action(s)
                s_next, r, done, _ = self.env.step(a)
                buffer_a.append(a[None, :])
                buffer_r.append((r + 8.0) / 8.0)
                if done:
                    break
                s = s_next
                if len(buffer_r) % BATCH_SIZE == 0 or it == EP_MAXLEN - 1:
                    discounted_r = []
                    last_value = self.model.value_estimate(s_next)
                    for r in buffer_r[::-1]:
                        last_value = r + GAMMA * last_value
                        discounted_r.append(last_value)
                    discounted_r.reverse()
                    batch_s = np.concatenate(buffer_s, axis=0)
                    batch_a = np.concatenate(buffer_a, axis=0)
                    batch_r = np.array(discounted_r, dtype=np.float32)[:, None]

                    # Block rolling event and release model update if queue size reaches maximum
                    ROLLING_EVENT.wait()
                    GLOBAL_QUEUE.put((batch_s, batch_a, batch_r))
                    if GLOBAL_QUEUE.qsize() >= MAX_QSIZE:
                        UPDATE_EVENT.set()
                        ROLLING_EVENT.clear()
                        print(datetime.now(), '--Iteration {} --Global queue filled')
                    # Clear buffer after model update
                    buffer_s.clear()
                    buffer_a.clear()
                    buffer_r.clear()
        print(' [*] Worker {} finish and exit'.format(self.wid))




if __name__ == '__main__':
    args = add_arguments()
    if args.method == 'kl_pen':
        METHOD = dict(name='kl_pen', kl_target=0.01, lam=0.5)
    elif args.method == 'clip':
        METHOD = dict(name='clip', epsilon=0.2)
    else:
        raise NotImplementedError

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    else:
        files = os.listdir(args.logdir)
        files = [os.path.join(args.logdir, fn) for fn in files]
        for f in files:
            os.remove(f)

    ppo = PPOModel(args.v_lr, args.pi_lr, S_DIM, A_DIM, method=METHOD, logdir=args.logdir)
    _, GLOBAL_COUNTER = load(ppo.sess, './ckpt/dppo/')

    processes = []
    for it in range(args.n_workers):
        worker = PPOWorker(it, model=ppo)
        worker.start()
        processes.append(worker)
    ppo.start()
    processes.append(ppo)

    env = gym.make('Pendulum-v0').unwrapped
    while not TERM_EVENT.is_set():
        discounted_r = 0.0
        running_gamma = 1.0
        s = env.reset()
        for it in range(300):
            env.render()
            s, r, done, info = env.step(ppo.choose_action(s))
            if done:
                break
            discounted_r += running_gamma * r
            running_gamma *= GAMMA
        print(datetime.now(), '--Total discounted reward ', discounted_r)

    print('Exit of all processes at ', datetime.now())