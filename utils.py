import tensorflow as tf
import numpy as np
import random
import os
import re


def hessian_vector_product(x, grad, variable):
    kl_grad_prod = tf.reduce_sum(grad * x)
    return tf.stop_gradient(tf.gradients(kl_grad_prod, variable)[0])


def build_conjugate_gradient(x, kl_grad, variable, n_iter=10, EPSILON = 1e-8,
                             func_Ax=hessian_vector_product):
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


def save(sess, model_path, model_name, global_step, remove_previous_files=True):
    saver = tf.train.Saver()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    elif len(os.listdir(model_path)) != 0 and remove_previous_files:
        fs = os.listdir(model_path)
        for f in fs:
            os.remove(os.path.join(model_path, f))

    saved_path = saver.save(sess, os.path.join(model_path, model_name), global_step=global_step)
    print(' [*] MODEL SAVED IN: ' + saved_path)
    return saved_path


def load(sess, model_path):
    print(" [*] Reading checkpoints...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_path, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def set_global_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)