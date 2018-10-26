#!/usr/bin/env python

"""
Code to load the policy learned from imitation learning and calculate reward
Example usage:
    python3 run_imitation.py Humanoid-v2-epoch30 Humanoid-v2 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import numpy as np
import tf_util
import gym

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


MODEL_DIR = 'saved_models'


def load_policy_from_file(sess, filename):
    graph = tf.get_default_graph()

    tf.saved_model.loader.load(
        sess,
        [tag_constants.SERVING],
        filename
    )
    input_placeholder = graph.get_tensor_by_name('input_placeholder:0')
    output = graph.get_tensor_by_name('fully_connected_2/BiasAdd:0')

    return load_policy_from_session(input_placeholder, output)


def load_policy_from_session(input_placeholder, output):
    return tf_util.function([input_placeholder], output)


def run_simulator(policy_fn, envname, num_rollouts, max_timesteps=None, render=False):
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    return_mean = np.mean(returns)
    return_std = np.std(returns)
    print("return mean: ", return_mean)
    print("return_std: ", return_std)

    return return_mean, return_std, np.array(observations)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('imitation_policy_file', type=str)
    parser.add_argument('--envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=40)
    args = parser.parse_args()
    if args.envname is None:
        args.envname = "-".join(args.imitation_policy_file.split("-")[:2])

    with tf.Session() as sess:
        print('loading and building imitation policy')
        policy_fn = load_policy_from_file(sess,
            os.path.join(MODEL_DIR, args.imitation_policy_file))
        print('loaded and built')

        _ = run_simulator(policy_fn, args.envname, args.num_rollouts,
            args.max_timesteps, args.render)

