import tensorflow as tf
import os
import re


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


