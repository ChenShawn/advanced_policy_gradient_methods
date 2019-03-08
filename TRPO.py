import tensorflow as tf
import gym
from gym.wrappers import Monitor
import argparse
import numpy as np
import os
from datetime import datetime

from utils import save, load, build_conjugate_gradient, AgentBase
from evaluate import AgentEvaluator


EPSILON = 1e-7
"""
The only difference between TRPO and TNPG is the training,
where TRPO uses a backtracking with exponential decay step sizes.
In this file we use the Pendulum-v0 environment.
"""

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=3000, help='Nnumber of total iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--ep_maxlen', type=int, default=200, help='max length of each episode')
    parser.add_argument('--v_lr', type=float, default=1e-4, help='learning rate of value function update')
    parser.add_argument('--gamma', type=float, default=0.9, help='discounted reward')
    parser.add_argument('--v_iter', type=int, default=5, help='number of iterations to train v')
    parser.add_argument('--pi_iter', type=int, default=4, help='number of iterations to train pi')
    parser.add_argument('--delta', type=float, default=1e-3, help='size of trust region')
    parser.add_argument('--model_dir', type=str, default='./ckpt/trpo/', help='model directory')
    parser.add_argument('--logdir', type=str, default='./logs/trpo/', help='log directory')
    parser.add_argument('--evaluate_every', type=int, default=100, help='number of iterations to evaluate agent')
    return parser.parse_args()


def build_gaussian_network(input_op, output_dim, scope, mu_scale=2.0, trainable=True, reuse=False):
    """build_gaussian_network
    :param output_dim: tye int representing the dimension of the output
    :return: two values, type (tf.distributions.Normal, list[trainable_vars])
    """
    kwargs = {'trainable': trainable, 'kernel_initializer': tf.orthogonal_initializer()}
    with tf.variable_scope(scope, reuse=reuse):
        h1 = tf.layers.dense(input_op, 128, activation=tf.nn.relu, **kwargs)
        mu = mu_scale * tf.layers.dense(h1, output_dim, activation=tf.nn.tanh, **kwargs)
        sigma = tf.layers.dense(h1, output_dim, activation=tf.nn.softplus, **kwargs)
        gaussian = tf.distributions.Normal(mu, sigma)
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return gaussian, train_vars


def collect_multi_batch(env, agent, maxlen, batch_size=64, qsize=2, gamma=0.9):
    """collect_multi_batch
    See collect_one_trajectory docstring
    :return: three lists of batch data (s, a, r)
    """
    que = []
    s_init = env.reset()
    que.append(s_init[None, :])
    for it in range(qsize - 1):
        st, r, done, _ = env.step(env.action_space.sample())
        que.append(st[None, :])
    # Interact with environment
    buffer_s, buffer_a, buffer_r = [], [], []
    for it in range(maxlen):
        env.render()
        s = np.concatenate(que, axis=-1)
        a = agent.choose_action(s)
        buffer_s.append(s)
        s, r, done, _ = env.step(a)
        que.pop(0)
        que.append(s[None, :])
        buffer_a.append(a[None, :])
        r = (r + 8.0) / 8.0
        buffer_r.append(r)
        if done:
            break
    # Accumulate rewards
    discounted_r = []
    last_value = agent.get_value(np.concatenate(que, axis=-1))
    for r in buffer_r[::-1]:
        last_value = r + gamma * last_value
        discounted_r.append(last_value)
    discounted_r.reverse()
    state_data, action_data, reward_data = [], [], []
    for it in range(0, maxlen, batch_size):
        if it >= len(buffer_s):
            break
        states_array = np.concatenate(buffer_s[it: it + batch_size], axis=0)
        actions_array = np.concatenate(buffer_a[it: it + batch_size], axis=0)
        rewards_array = np.array(discounted_r[it: it + batch_size], dtype=np.float32)[:, None]
        # rewards_array = np.clip(rewards_array, -1.0, 5.0)
        state_data.append(states_array)
        action_data.append(actions_array)
        reward_data.append(rewards_array)
    return state_data, action_data, reward_data



class TRPOModel(AgentBase):
    def __init__(self, v_lr, model_dir, delta=1e-3):
        self.state = tf.placeholder(tf.float32, [None, 6], name='state')
        self.action = tf.placeholder(tf.float32, [None, 1], name='action')
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
        self.alpha = tf.placeholder(tf.float32, name='alpha')
        self.delta = delta

        # Advantage function definition
        print(' [*] Building advantage function...')
        kwargs = {'kernel_initializer': tf.orthogonal_initializer()}
        with tf.variable_scope('value'):
            h1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu, name='h1', **kwargs)
            self.value = tf.layers.dense(h1, 1, activation=None, name='value', **kwargs)
            self.advantage = self.reward - self.value

            self.v_loss = tf.reduce_mean(tf.square(self.advantage))
        v_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')
        self.v_train = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss, var_list=v_vars)

        # Policy function definition
        print(' [*] Building policy function...')
        self.policy, pi_vars = build_gaussian_network(self.state, 1, scope='policy',
                                                      trainable=False, reuse=False)
        old_policy, old_vars = build_gaussian_network(self.state, 1, scope='old_policy',
                                                      trainable=True, reuse=False)
        ref_policy, ref_vars = build_gaussian_network(self.state, 1, scope='old_policy',
                                                      trainable=False, reuse=True)
        with tf.name_scope('policy_ops'):
            self.assign_old = [old.assign(new) for old, new in zip(old_vars, pi_vars)]
            self.sample_op = self.policy.sample(1)
        with tf.name_scope('surrogate_loss'):
            ratio = old_policy.prob(self.action) / ref_policy.prob(self.action)
            surrogate = ratio * self.advantage
            self.pi_loss_old = -tf.reduce_mean(surrogate)

            ratio = self.policy.prob(self.action) / old_policy.prob(self.action)
            surrogate = ratio * self.advantage
            self.pi_loss_new = -tf.reduce_mean(surrogate)

        # Convert Adam gradient to natural gradient
        print(' [*] Building natural gradient...')
        with tf.variable_scope('policy_optim'):
            self.kl = tf.distributions.kl_divergence(old_policy, self.policy)
            kl_zero = tf.distributions.kl_divergence(ref_policy, old_policy)
            pi_grads = tf.gradients(self.pi_loss_old, old_vars)
            kl_grads = tf.gradients(kl_zero, old_vars)

            self.assign_new = []
            for grad, kl_grad, var, nvar in zip(pi_grads, kl_grads, old_vars, pi_vars):
                conj = build_conjugate_gradient(grad, kl_grad, var)
                nat_grad = tf.sqrt((2.0 * delta) / (tf.reduce_sum(grad * conj) + EPSILON)) * conj
                new_value = var - self.alpha * nat_grad
                self.assign_new.append(nvar.assign(new_value))

        # Summaries definition
        print(' [*] Building summaries...')
        model_variance = tf.reduce_mean(self.policy._scale)
        self.sums = tf.summary.merge([
            tf.summary.scalar('max_rewards', tf.reduce_max(self.reward)),
            tf.summary.scalar('mean_advantage', tf.reduce_mean(self.advantage)),
            tf.summary.scalar('pi_loss_old', self.pi_loss_old),
            tf.summary.scalar('pi_loss_new', self.pi_loss_new),
            tf.summary.scalar('v_loss', self.v_loss),
            tf.summary.scalar('model_variance', model_variance)
        ], name='summaries')

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Model built finished')
        _, self.counter = load(self.sess, model_dir)


    def choose_action(self, s):
        a = self.sess.run(self.sample_op, feed_dict={self.state: s})
        return a[0, :, 0]


    def line_search(self, s, a, r, max_iter=10):
        alpha = 1.0
        feed_dict = {self.state: s, self.action: a, self.reward: r}
        for it in range(max_iter):
            feed_dict[self.alpha] = alpha
            self.sess.run(self.assign_new, feed_dict=feed_dict)
            kl, old, new = self.sess.run([self.kl, self.pi_loss_old, self.pi_loss_new], feed_dict=feed_dict)
            if np.all(kl < self.delta) and new < old:
                self.sess.run(self.assign_old)
                info = 'Global-step {} --Accept update with alpha={}, kl={}, old={}, new={}'
                print(info.format(self.counter, alpha, kl.mean(), old, new))
                return alpha
            else:
                alpha *= 0.5
        return 0.0


    def update(self, s, a, r, v_iter=3, pi_iter=2, writer=None, counter=0):
        feed_dict = {self.state: s, self.action: a, self.reward: r}
        # update policy
        for _ in range(pi_iter):
            self.line_search(s, a, r)

        # update value function
        for _ in range(v_iter):
            self.sess.run(self.v_train, feed_dict=feed_dict)
        if writer is not None:
            sumstr = self.sess.run(self.sums, feed_dict=feed_dict)
            writer.add_summary(sumstr, counter)


    def get_value(self, s):
        return self.sess.run(self.value, feed_dict={self.state: s})[0, 0]



if __name__ == '__main__':
    env_name = 'Pendulum-v0'

    args = add_arguments()
    if not os.path.exists(args.model_dir) or not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.logdir) or not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    env = gym.make(env_name).unwrapped
    recoder = Monitor(env, directory='./logs/records/' + env_name, resume=True,
                      video_callable=lambda x: x > 0)
    recoder._max_episode_steps = args.ep_maxlen

    STATE_SHAPE = env.observation_space.shape
    ACTION_SHAPE = env.action_space.shape

    model = TRPOModel(args.v_lr, model_dir=args.model_dir, delta=args.delta)
    writer = tf.summary.FileWriter(args.logdir, model.sess.graph)
    evaluator = AgentEvaluator()

    # Start to train
    model.sess.run(model.assign_old)
    for it in range(args.iter):
        slist, alist, rlist = collect_multi_batch(env, model, maxlen=args.ep_maxlen,
                                                  batch_size=args.batch_size)
        for s, a, r in zip(slist, alist, rlist):
            model.update(s, a, r, v_iter=args.v_iter, pi_iter=args.pi_iter,
                         writer=writer, counter=model.counter)
            model.counter += 1
        if it % args.evaluate_every == 10:
            evaluator.evaluate(env, model, maxlen=args.ep_maxlen)
            # evaluator.record_video(recoder, model, maxlen=args.ep_maxlen)

    save(model.sess, './ckpt/trpo/', env_name, model.counter)
    evaluator.to_csv(os.path.join('./logs/records/' + env_name, 'trpo.csv'))