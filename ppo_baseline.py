import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import argparse
import os

from utils import save, load


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=50, help='number of total iteration')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--method', type=str, default='clip', help='either kl_pen or clip')
    parser.add_argument('--ep_maxlen', type=int, default=200, help='max length of each episode')
    parser.add_argument('--v_lr', type=float, default=2e-4, help='learning rate of value function update')
    parser.add_argument('--pi_lr', type=float, default=1e-4, help='learning rate of policy function')
    parser.add_argument('--gamma', type=float, default=0.9, help='discounted reward')
    parser.add_argument('--v_iter', type=int, default=10, help='number of iterations to train v')
    parser.add_argument('--pi_iter', type=int, default=10, help='number of iterations to train pi')
    parser.add_argument('--model_dir', type=str, default='./ckpt/ppo/', help='model directory')
    parser.add_argument('--logdir', type=str, default='./logs/ppo/', help='log directory')
    return parser.parse_args()


class PPO(object):
    def __init__(self, v_lr, pi_lr, s_dim, a_dim, method, model_dir):
        self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
        self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        self.method = method

        # Critic (Using advantage function to reduce variance)
        with tf.variable_scope('value'):
            l1 = tf.layers.dense(self.tfs, 100, activation=tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.advantage = self.tfdc_r - self.v
            self.v_loss = tf.reduce_mean(tf.square(self.advantage))
            v_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')

        # Actor (Function pi(a|s) following behavior policy)
        pi, pi_vars = self._build_policy('pi', a_dim, trainable=True)
        oldpi, oldpi_vars = self._build_policy('oldpi', a_dim, trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_vars, oldpi_vars)]

        # self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.tfadv = tf.stop_gradient(self.advantage)
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
            if method['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                self.pi_loss = -tf.reduce_mean(surr - self.tflam * kl)
            elif method['name'] == 'clip':
                self.pi_loss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.0 - method['epsilon'], 1.0 + method['epsilon']) * self.tfadv)
                )
            else:
                raise NotImplementedError

        with tf.variable_scope('train_ops'):
            self.vtrain_op = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss, var_list=v_vars)
            self.pitrain_op = tf.train.AdamOptimizer(pi_lr).minimize(self.pi_loss, var_list=pi_vars)

        self.sums = tf.summary.merge([
            tf.summary.scalar('mean_reward', tf.reduce_mean(self.tfdc_r)),
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
        _, self.counter = load(self.sess, model_dir)


    def update(self, s, a, r, v_iter=10, pi_iter=10, writer=None):
        self.sess.run(self.update_oldpi_op)
        feed_dict = {self.tfs: s, self.tfa: a, self.tfdc_r: r}
        if self.method['name'] == 'kl_pen':
            feed_dict[self.tflam] = self.method['lam']
            for _ in range(pi_iter):
                _, kl = self.sess.run([self.pitrain_op, self.kl_mean], feed_dict=feed_dict)
                if kl > 4 * self.method['kl_target']:
                    break
            if kl < self.method['kl_target'] / 1.5:
                self.method['lam'] *= 0.5
            elif kl > self.method['kl_target'] * 1.5:
                self.method['lam'] *= 2.0
            self.method['lam'] = np.clip(self.method['lam'], 1e-4, 10)
        else:
            for _ in range(pi_iter):
                self.sess.run(self.pitrain_op, feed_dict=feed_dict)
        for _ in range(v_iter):
            self.sess.run(self.vtrain_op, feed_dict=feed_dict)
        if writer is not None:
            sumstr = self.sess.run(self.sums, feed_dict=feed_dict)
            writer.add_summary(sumstr, self.counter)
        self.counter += 1


    def _build_policy(self, name, n_out, trainable=True, reuse=False):
        # params = {'trainable': trainable, 'kernel_initializer': tf.orthogonal_initializer()}
        params = {'trainable': trainable}
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, **params)
            mu = 2.0 * tf.layers.dense(l1, n_out, tf.nn.tanh, **params)
            sigma = tf.layers.dense(l1, n_out, tf.nn.softplus, **params)
            gaussian = tf.distributions.Normal(loc=mu, scale=sigma)
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return gaussian, vars


    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, feed_dict={self.tfs: s})[0]
        return np.clip(a, -2.0, 2.0)


    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, feed_dict={self.tfs: s})[0, 0]



if __name__ == '__main__':
    env_name = 'Pendulum-v0'
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

    env = gym.make(env_name).unwrapped
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]

    ppo = PPO(args.v_lr, args.pi_lr, S_DIM, A_DIM, method=METHOD, model_dir=args.model_dir)
    all_ep_r = []
    writer = tf.summary.FileWriter(args.logdir, ppo.sess.graph)

    for ep in range(args.iter):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(args.ep_maxlen):
            env.render()
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8.0) / 8.0)
            s = s_
            ep_r += r

            # update ppo
            if (t + 1) % args.batch_size == 0 or t == args.ep_maxlen - 1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + args.gamma * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br, v_iter=args.v_iter, pi_iter=args.pi_iter, writer=writer)
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print(
            'Ep: %i' % ep,
            "| Ep_r: %i" % ep_r,
            ("| Lam: %.4f" % ppo.method['lam']) if ppo.method['name'] == 'kl_pen' else '',
        )
    save(ppo.sess, './ckpt/ppo/', env_name, ppo.counter)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()