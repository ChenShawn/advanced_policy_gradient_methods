import tensorflow as tf
import gym
from gym.wrappers import Monitor
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import os
from queue import Queue

from utils import save, load
EPSILON = 1e-10


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', type=int, default=3000, help='Nnumber of total iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--v_lr', type=float, default=2e-4, help='learning rate of value function update')
    parser.add_argument('--pi_lr', type=float, default=1e-4, help='learning rate of policy update')
    parser.add_argument('--gamma', type=float, default=0.9, help='discounted reward')
    parser.add_argument('--ep_maxlen', type=int, default=2000, help='maximum length of one episode')
    parser.add_argument('--v_iter', type=int, default=5, help='number of iterations to train v')
    parser.add_argument('--pi_iter', type=int, default=5, help='number of iterations to train pi')
    parser.add_argument('--delta', type=float, default=1e-3, help='size of trust region')
    parser.add_argument('--model_dir', type=str, default='./ckpt/tnpg/', help='model directory')
    parser.add_argument('--logdir', type=str, default='./logs/tnpg/', help='log directory')
    parser.add_argument('--evaluate_every', type=int, default=50, help='number of iterations to evaluate agent')
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
    return tf.stop_gradient(tf.gradients(kl_grad_prod, variable)[0])


def build_conjugate_gradient(x, kl_grad, variable, n_iter=10, func_Ax=hessian_vector_product):
    """build_conjugate_gradient
    :param x: type tf.Tensor, the initial value of x
    :param kl_grad: type tf.Tensor, the gradient of the objective
    :param variable: type tf.Variable
    :return: the converged conjugate gradient vector \tilde{x} = H^{-1}x

    Truncated natural policy gradient uses fixed number of iterations in the inner loop
    Reference: https://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    x = tf.stop_gradient(x)
    r = x - func_Ax(x, kl_grad, variable)
    p = tf.stop_gradient(tf.identity(r))
    r_dot_r = tf.reduce_sum(tf.square(r))
    for k in range(n_iter):
        Ap = func_Ax(p, kl_grad, variable)
        p_dot_Ap = tf.reduce_sum(p * Ap)
        alpha = r_dot_r / (p_dot_Ap + EPSILON)
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_r_new = tf.reduce_sum(tf.square(r))
        beta = r_dot_r_new / (r_dot_r + EPSILON)
        r_dot_r = r_dot_r_new
        p = r + beta * p
    return x


def collect_multi_batch(env, agent, maxlen, batch_size=64, qsize=5):
    """collect_multi_batch
    See collect_one_trajectory docstring
    :return: three lists of batch data (s, a, r)
    """
    que = []
    s_init = env.reset()
    que.append(s_init[None, :])
    for it in range(qsize - 1):
        st, r, done, _ = env.step([-0.99])
        que.append(st[None, :])
    # Interact with environment
    buffer_s, buffer_a, buffer_r = [], [], []
    for it in range(maxlen):
        # The idea works on Atari games
        # if normalize_state:
        #     s = np.clip((s - s.mean()) / s.std(), -5.0, 5.0)
        s = np.concatenate(que, axis=-1)
        a = agent.choose_action(s)
        buffer_s.append(s)
        s, r, done, _ = env.step(a)
        que.pop(0)
        que.append(s[None, :])
        buffer_a.append(a[None, :])
        r = (r + 0.3) * 2.0
        buffer_r.append(r)
        if done:
            break
    # Accumulate rewards
    discounted = 1.0
    for it in range(len(buffer_r) - 2, -1, -1):
        buffer_r[it] = buffer_r[it + 1] + discounted * buffer_r[it]
        discounted *= args.gamma
    state_data, action_data, reward_data = [], [], []
    for it in range(0, maxlen, batch_size):
        if it >= len(buffer_s):
            break
        states_array = np.concatenate(buffer_s[it: it + batch_size], axis=0)
        actions_array = np.concatenate(buffer_a[it: it + batch_size], axis=0)
        rewards_array = np.array(buffer_r[it: it + batch_size], dtype=np.float32)[:, None]
        # rewards_array = np.clip(rewards_array, -1.0, 5.0)
        state_data.append(states_array)
        action_data.append(actions_array)
        reward_data.append(rewards_array)
    return state_data, action_data, reward_data


class TNPGModel(object):
    def __init__(self, v_lr, pi_lr, model_dir, delta=1e-3):
        self.state = tf.placeholder(tf.float32, [None, 10], name='state')
        self.action = tf.placeholder(tf.float32, [None, 1], name='action')
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')

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
        self.policy, pi_vars = build_gaussian_network(self.state, 1, scope='policy')
        old_policy, old_vars = build_gaussian_network(self.state, 1, scope='policy', trainable=False, reuse=True)
        with tf.name_scope('policy_ops'):
            # self.assign_op = [old.assign(new) for old, new in zip(old_vars, pi_vars)]
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
                nat_grad = tf.sqrt((2.0 * delta) / (tf.reduce_sum(grad * conj) + EPSILON)) * conj
                conj_grads.append((nat_grad, var))
            self.pi_train = optim.apply_gradients(conj_grads)

        # Summaries definition
        print(' [*] Building summaries...')
        model_variance = tf.reduce_mean(self.policy._scale)
        self.sums = tf.summary.merge([
            tf.summary.scalar('max_rewards', tf.reduce_max(self.reward)),
            tf.summary.scalar('mean_advantage', tf.reduce_mean(self.advantage)),
            tf.summary.scalar('pi_loss', self.pi_loss),
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


    def update(self, s, a, r, v_iter, pi_iter, writer=None, counter=0):
        feed_dict = {self.state: s, self.action: a, self.reward: r}
        # self.sess.run(self.assign_op)
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

    def evaluate(self, env, agent, num_episode=10, gamma=0.9, maxlen=200, qsize=5):
        ans = []
        for it in range(num_episode):
            acc_r = 0.0
            coeff = 1.0
            s0 = env.reset()
            que = [s0[None, :]]
            for it in range(qsize - 1):
                st, r, done, info = env.step(env.action_space.sample())
                que.append(st[None, :])
            for jt in range(maxlen):
                env.render()
                s = np.concatenate(que, axis=-1)
                a = agent.choose_action(s)
                st, r, done, _ = env.step(a)
                que.pop(0)
                que.append(st[None, :])
                acc_r += coeff * r
                coeff *= gamma
                if done:
                    break
            ans.append(acc_r)
        print('Total reward in global step {}: {}'.format(agent.counter, ans[-1]))
        self.records.append(ans)

    def record_video(self, env, agent, norm_state=False):
        """record_video
        :param env: should be a gym.Env wrapped by Monitor
        :param record_dir: where to save the video
        """
        s = env.reset()
        while True:
            env.render()
            if norm_state:
                s = np.clip((s - s.mean()) / s.std(), -5.0, 5.0)
            a = agent.choose_action(s)
            s, r, done, _ = env.step(a)
            if done:
                break

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
    recoder = Monitor(env, directory='./logs/records/' + env_name, resume=True,
                      video_callable=lambda x: x % args.evaluate_every == 10)
    recoder._max_episode_steps = 2 * args.ep_maxlen

    # Definition of global variables
    A_DIM = env.action_space.shape[0]
    A_LOWER = env.action_space.low[0]
    A_HIGHER = env.action_space.high[0]

    model = TNPGModel(args.v_lr, args.pi_lr, args.model_dir, delta=args.delta)
    writer = tf.summary.FileWriter(args.logdir, model.sess.graph)
    evaluator = AgentEvaluator()
    for it in range(args.iter):
        # The trajectory obtained via this way would be too long for a batch of input
        # s, a, r = collect_one_trajectory(env, model, args.ep_maxlen)
        slist, alist, rlist = collect_multi_batch(env, model, maxlen=args.ep_maxlen,
                                                  batch_size=args.batch_size)
        for s, a, r in zip(slist, alist, rlist):
            model.update(s, a, r, v_iter=args.v_iter, pi_iter=args.pi_iter,
                         writer=writer, counter=model.counter)
            model.counter += 1
        if it % args.evaluate_every == 1:
            evaluator.evaluate(env, model, maxlen=args.ep_maxlen)
            # evaluator.record_video(recoder, model)

    print(model.sess)
    save(model.sess, './ckpt/tnpg/', env_name, model.counter)
    evaluator.to_csv(os.path.join('./logs/records/' + env_name, 'tnpg.csv'))
