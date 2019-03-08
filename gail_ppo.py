import tensorflow as tf
import gym
from tensorflow.contrib import slim
from gym.wrappers import Monitor
import argparse
import numpy as np
import os
import random
import threading

from utils import save, load, set_global_seed, AgentBase, TrajectoryProcessor
from evaluate import AgentEvaluator


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--what', type=str, default='sample')
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



if __name__ == '__main__':
    processor = TrajectoryProcessor()
    processor.load('./logs/records/Pendulum-v0/expert.hdf5')
    print(processor.data)