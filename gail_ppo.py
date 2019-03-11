import tensorflow as tf
import gym
from tensorflow.contrib import slim
import matplotlib.pyplot as plt
import argparse
import random
import matplotlib.style as mstyle
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils import save, load, set_global_seed, AgentBase, TrajectoryProcessor


S_DIM, A_DIM, A_SCALE = 3, 1, 2.0
BATCH_SIZE = 64
GAMMA = 0.9
LEARNING_RATE = 5e-4
MODEL_DIR = './ckpt/gail/'
LOGDIR = './logs/gail/'
EVALUATE_EVERY = 100
EP_MAXLEN = 200


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--what', type=str, default='clone', help='do what')
    parser.add_argument('-i', '--iter', type=int, default=5000, help='Nummber of iterations')
    return parser.parse_args()


class BehaviorClone(AgentBase):
    def __init__(self):
        self.state = tf.placeholder(tf.float32, [None, S_DIM], name='state')
        self.action = tf.placeholder(tf.float32, [None, A_DIM], name='action')
        self.policy, pi_vars = self._build_policy()

        loss = tf.losses.mean_squared_error(labels=self.action, predictions=self.policy)
        self.optim = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, var_list=pi_vars)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)


    def _build_policy(self):
        with tf.variable_scope('Policy'):
            h1 = tf.layers.dense(self.state, 64, activation=tf.nn.relu, name='h1')
            h2 = tf.layers.dense(h1, 32, activation=tf.nn.relu, name='h2')
            h3 = A_SCALE * tf.layers.dense(h2, 1, activation=tf.tanh, name='h3')
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')
        return h3, train_vars


    def train(self, iteration, state, action):
        env = gym.make('Pendulum-v0').unwrapped
        rewards = []
        for it in tqdm(range(iteration)):
            indices = random.randint(0, min(len(state), len(action)))
            s, a = state[indices], action[indices]
            self.sess.run(self.optim, feed_dict={self.state: s, self.action: a})
            r = self.evaluate(env, render=it > 800)
            rewards.append(r)
        print(' [*] Training finished... Ready to save...')
        return rewards


    def evaluate(self, env, render=False):
        s = env.reset()
        reward_buffer = []
        for it in range(EP_MAXLEN):
            if render:
                env.render()
            s, r, done, info = env.step(self.choose_action(s))
            r = (r + 8.0) / 8.0
            reward_buffer.append(r)
            if done:
                break
        discounted_r = 0.0
        for r in reward_buffer[::-1]:
            discounted_r = r + GAMMA * discounted_r
        return discounted_r


    def choose_action(self, s):
        return self.sess.run(self.policy, feed_dict={self.state: s[None, :]})[0]



if __name__ == '__main__':
    processor = TrajectoryProcessor()
    processor.load('./logs/records/Pendulum-v0/expert.hdf5')

    args = add_arguments()
    if args.what == 'clone':
        model = BehaviorClone()
        rewards = model.train(args.iter, processor.data['state'], processor.data['action'])

        df = pd.DataFrame(rewards)
        mu = df.ewm(span=40).mean().values[:, 0]
        sigma = df.ewm(span=40).std().values[:, 0]
        sigma[0] = 0.0

        mstyle.use('ggplot')
        plt.figure()
        plt.plot(mu, color='orange')
        plt.fill_between(np.arange(len(mu)), mu + sigma, mu - sigma, color='orange', alpha=0.2)
        plt.xlabel('Iterations')
        plt.ylabel('Average rewards')
        plt.title('Training of Imitation Learning')
        plt.show()

    elif args.what == 'gail':
        pass

    else:
        raise NotImplementedError