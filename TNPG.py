import tensorflow as tf
import gym
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import os

from utils import save, load


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=3000, help='Number of total iterations')
    parser.add_argument('--v_lr', type=float, default=2e-4, help='Learning rate of value function update')
    parser.add_argument('--pi_lr', type=float, default=1e-4, help='Learning rate of policy update')
    parser.add_argument('--gamma', type=float, default=0.9, help='discounted reward')
    parser.add_argument('--ep_maxlen', type=int, default=200, help='the maximum length of one episode')
    parser.add_argument('--v_iter', type=int, default=10, help='number of iterations to train v')
    parser.add_argument('--pi_iter', type=int, default=10, help='number of iterations to train pi')
    parser.add_argument('--model_dir', type=str, default='./ckpt/tnpg/', help='model directory')
    parser.add_argument('--logdir', type=str, default='./logs/tnpg/', help='log directory')
    parser.add_argument('--evaluate_every', type=int, default=20, help='number of iterations to evaluate agent')
    return parser.parse_args()


def build_gaussian_network(input_op, output_dim, scope, mu_scale=1.0, trainable=True, reuse=False):
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


def hessian_vector_product(x, grad, variable):
    kl_grad_prod = tf.reduce_sum(grad * x)
    return tf.gradients(kl_grad_prod, variable)[0]


def build_conjugate_gradient(x, kl_grad, variable, n_iter=10, func_Ax=hessian_vector_product):
    """build_conjugate_gradient
    :param x: type tf.Tensor, the initial value of x
    :param kl_grad: type tf.Tensor, the gradient of the objective
    :param variable: type tf.Variable
    :return: the converged conjugate gradient vector \tilde{x} = H^{-1}x

    Truncated natural policy gradient uses fixed number of iterations in the inner loop
    Reference: https://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    r = x - func_Ax(x, kl_grad, variable)
    p = tf.identity(r)
    r_dot_r = tf.reduce_sum(tf.square(r), axis=[1, 2])
    for k in range(n_iter):
        p_dot_Ap = tf.reduce_sum(p * func_Ax(p, kl_grad, variable), axis=[1, 2])
        alpha = r_dot_r / p_dot_Ap
        x = x + alpha * p
        r = r - alpha * func_Ax(p, kl_grad, variable)
        r_dot_r_new = tf.reduce_sum(tf.square(r), axis=[1, 2])
        beta = r_dot_r_new / r_dot_r
        r_dot_r = r_dot_r_new
        p = r + beta * p
    return x


def collect_one_trajectory(env, agent, maxlen, normalize_state=False, normalize_reward=False):
    """collect_one_trajectory
    :param env: type gym.Env registered
    :param agent: type TNPGModel
    :param maxlen: maximum length of each trajectory
    :param normalize_state: whether normalizing observations
    :param normalize_reward: whether normalizing rewards
    :return: triplet (s, a, r) of type np.array
    """
    s = env.reset()
    # Interact with environment
    buffer_s, buffer_a, buffer_r = [], [], []
    for it in range(maxlen):
        if normalize_state:
            s = np.clip((s - s.mean()) / s.std(), -5.0, 5.0)
        a = agent.choose_action(s)
        buffer_s.append(s[None, :])
        s, r, done, _ = env.step(a)
        buffer_a.append(a[None, :])
        if normalize_reward:
            r = (r - 8.0) / 8.0
        buffer_r.append(r)
        if done:
            break
    # Accumulate rewards
    gamma = 1.0
    for it in range(len(buffer_r) - 2, -1, -1):
        buffer_r[it] = buffer_r[it + 1] + gamma * buffer_r[it]
        gamma *= args.gamma
    states_array = np.concatenate(buffer_s, axis=0)
    actions_array = np.concatenate(buffer_a, axis=0)
    rewards_array = np.array(buffer_r, dtype=np.float32)[:, None]
    return states_array, actions_array, rewards_array


class TNPGModel(object):
    def __init__(self, v_lr, pi_lr, model_dir):
        self.state = tf.placeholder(tf.float32, [None, 2], name='state')
        self.action = tf.placeholder(tf.float32, [None, 1], name='action')
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')

        # Advantage function definition
        print(' [*] Building advantage function...')
        with tf.variable_scope('value'):
            h1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu, name='h1')
            self.value = tf.layers.dense(h1, 1, activation=None, name='value')
            self.advantage = self.reward - self.value

            self.v_loss = tf.reduce_mean(tf.square(self.advantage))
        v_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')
        self.v_train = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss, var_list=v_vars)

        # Policy function definition
        print(' [*] Building policy function...')
        self.policy, pi_vars = build_gaussian_network(self.state, A_DIM, scope='policy')
        old_policy, old_vars = build_gaussian_network(self.state, A_DIM, scope='old_policy', trainable=False)
        with tf.name_scope('policy_ops'):
            self.assign_op = [old.assign(new) for old, new in zip(old_vars, pi_vars)]
            self.sample_op = self.policy.sample(1)
        with tf.name_scope('surrogate_loss'):
            ratio = self.policy.prob(self.action) / old_policy.prob(self.action)
            surrogate = ratio * self.advantage
            self.pi_loss = -tf.reduce_mean(surrogate)

        # Convert Adam gradient to natural gradient
        print(' [*] Building natural gradient...')
        with tf.variable_scope('policy_optim'):
            kl = tf.distributions.kl_divergence(old_policy, self.policy)
            optim = tf.train.AdamOptimizer(pi_lr)
            pi_grads_and_vars = optim.compute_gradients(surrogate, var_list=pi_vars)
            pi_grads = [pair[0] for pair in pi_grads_and_vars]
            kl_grads = tf.gradients(kl, pi_vars)

            conj_grads = []
            for grad, kl_grad, var in zip(pi_grads, kl_grads, pi_vars):
                conj = build_conjugate_gradient(grad, kl_grad, var)
                conj_grads.append((conj, var))
            self.pi_train = optim.apply_gradients(conj_grads)

        # Summaries definition
        print(' [*] Building summaries...')
        model_variance = tf.reduce_mean(self.policy._scale)
        self.sums = tf.summary.merge([
            tf.summary.scalar('rewards', self.reward),
            tf.summary.scalar('advantage', self.advantage),
            tf.summary.scalar('pi_loss', self.pi_loss),
            tf.summary.scalar('v_loss', self.v_loss),
            tf.summary.scalar('model_variance', model_variance)
        ], name='summaries')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Model built finished')
        _, self.counter = load(self.sess, model_dir)


    def choose_action(self, s):
        a = self.sess.run(self.sample_op, feed_dict={self.state: s})
        return a[0]


    def update(self, s, a, r, v_iter, pi_iter, writer=None, counter=0):
        feed_dict = {self.state: s, self.action: a, self.reward: r}
        self.sess.run(self.assign_op)
        # update policy
        for _ in range(pi_iter):
            self.sess.run(self.pi_train, feed_dict=feed_dict)
        # update value function
        for _ in range(v_iter):
            self.sess.run(self.v_train, feed_dict=feed_dict)
        if writer is not None:
            sumstr = self.sess.run(self.sums, feed_dict=feed_dict)
            writer.add_summary(sumstr, counter)


    def async_update(self, sess, v_iter, pi_iter):
        pass


class AgentEvaluator(object):
    records = []

    def evaluate(self, env, agent, num_episode=10, gamma=0.9, maxlen=200, norm_state=False):
        s = env.reset()
        ans = []
        for it in range(num_episode):
            acc_r = 0.0
            coeff = 1.0
            for jt in range(maxlen):
                env.render()
                if norm_state:
                    s = np.clip((s - s.mean()) / s.std(), -5.0, 5.0)
                a = agent.choose_action(s)
                s, r, done, _ = env.step(a)
                acc_r += coeff * r
                coeff *= gamma
                if done:
                    break
            ans.append(acc_r)
        self.records.append(ans)

    def to_csv(self, csv_dir):
        df = pd.DataFrame()
        array = np.array(self.records, dtype=np.float32)
        info = list(map(lambda ll: ', '.join(ll), self.records))
        df['step'] = np.arange(len(self.records), dtype=np.int32)
        df['reward_mean'] = array.mean(axis=-1)
        df['reward_std'] = array.std(axis=-1)
        df['reward_smooth'] = df['reward_mean'].ewm(span=20).mean()
        df['upper_bound'] = df['reward_mean'] + df['reward_std']
        df['lower_bound'] = df['reward_mean'] - df['reward_std']
        df['info'] = pd.DataFrame(info)
        df.to_csv(csv_dir)

    def plot_from_csv(self, csv_dir, savefig=None):
        df = pd.read_csv(csv_dir)

        mstyle.use('ggplot')
        plt.figure()
        plt.plot(np.arange(df['reward_mean'].__len__()), df['reward_smooth'], color='orange')
        plt.fill_between(np.arange(df['reward_smooth'].__len__()),
                         df['upper_bound'],
                         df['lower_bound'],
                         color='orange', alpha=0.2)
        plt.ylabel('Averaged reward')
        plt.xlabel('Steps of env interaction')
        plt.title('TNPG on {}'.format(csv_dir.split('.')[0]))
        if savefig is not None and type(savefig) is str:
            plt.savefig(savefig, format='svg')
        plt.show()



if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    args = add_arguments()
    env = gym.make(env_name)

    # Definition of global variables
    A_DIM = env.action_space.shape[0]
    A_LOWER = env.action_space.low[0]
    A_HIGHER = env.action_space.high[0]

    model = TNPGModel(args.v_lr, args.pi_lr, args.model_dir)
    writer = tf.summary.FileWriter(args.logdir, model.sess.graph)
    evaluator = AgentEvaluator()
    for it in range(args.iter):
        s, a, r = collect_one_trajectory(env, model, args.ep_maxlen)
        model.update(s, a, r, v_iter=args.v_iter, pi_iter=args.pi_iter, writer=writer)
        if it % args.evaluate_every == (args.evaluate_every - 1):
            evaluator.evaluate(env, model)

    evaluator.to_csv(os.path.join('./logs/records/' + env_name, 'TNPG.csv'))