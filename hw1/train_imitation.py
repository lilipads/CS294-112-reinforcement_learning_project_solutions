"""
Code to train imitation learning model based on rollouts from expert policy.
Example Usage:
    python3 imitation_model.py Humanoid-v2.pkl --epoch 20
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import os
import argparse
import math
import numpy as np
import shutil
import run_imitation
import load_policy

DATA_DIR = 'expert_data'
OUTPUT_DIR = 'saved_models'


def prepare_data(filename):
    """
    load data from file and split into training and testing.

    filename: filename fo the pickle file that stores the observations and
        actions from rollouts after running the expert policy. The pickle
        file store a dictionary, with keys 'observation' and 'actions'
    
    return:
        X_train, X_test, y_train, y_tests: as list. X is observation space data,
            y is action space data.

    """
    with open(os.path.join(DATA_DIR, filename), 'rb') as f:
        data = pickle.loads(f.read())
    data['actions'] = np.squeeze(data['actions'], axis=1)
    return _shuffle_and_split_data(data)


def _shuffle_and_split_data(data):
    """
    data is a json object that stores observations and actions

    """
    X = np.array(data['observations'])
    y = data['actions']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def build_model(input_placeholder, label_placeholder, output_dim):
    """
    build a 2 64-unit hidden layer feedforward neural network.

    input:
        input_placeholder: tf placeholder for inputs
        label_placeholder: tf placeholder for labels
        output_dim: dimension of the output (action space)
    return:
        output: tensor for the last layer
        train_op: tensor for the optimization option
        mse_loss: tensor for the loss function

    """    
    fc1 = tf.contrib.layers.fully_connected(input_placeholder, 64,
        weights_regularizer=tf.contrib.layers.l2_regularizer(args.regularization_weight),
        activation_fn=tf.tanh)
    fc2 = tf.contrib.layers.fully_connected(fc1, 64,
        weights_regularizer=tf.contrib.layers.l2_regularizer(args.regularization_weight),
        activation_fn=tf.tanh)
    output = tf.contrib.layers.fully_connected(fc2, output_dim, activation_fn=None,
        weights_regularizer=tf.contrib.layers.l2_regularizer(args.regularization_weight))
    
    mse_loss = tf.losses.mean_squared_error(
        label_placeholder,
        output,
        loss_collection=tf.GraphKeys.LOSSES
    )
    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(mse_loss)
    return output, train_op, mse_loss


def prepare_dagger_data(input_placeholder, output):
    """
    apply expert policy on observations generated by imitation policy.

    Return:
        X_train, X_test, y_train, y_test: shuffled training and test data set
            on the imitation policy rollouts. The features are observations generated
            by the imtiation policy rollouts. The labels are actions taken by the experts
            on these observations.
    """
    # load policy
    imitation_policy_fn = run_imitation.load_policy_from_session(input_placeholder, output)
    # run simulation with this policy
    return_mean, return_std, imitation_observations\
         = run_imitation.run_simulator(imitation_policy_fn, args.envname, 
                                       args.dagger_num_rollouts)
    data = {'observations': imitation_observations}
    # generate rollout by querying the expert policy
    expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
    data['actions'] = expert_policy_fn(imitation_observations)
    # return train and test data
    return return_mean, return_std, _shuffle_and_split_data(data)


def run():
    """
    load data and model, then train the model in a number of epochs
    
    """
    X_train, X_test, y_train, y_test = prepare_data(args.training_data_file)

    batch_count = int(math.ceil(len(X_train) / args.batch_size))
    input_dim = len(X_train[0])
    output_dim = len(y_train[0])

    return_means = []
    return_stds = []

    with tf.Session() as sess:
        input_placeholder = tf.placeholder(tf.float32, [None, input_dim], name="input_placeholder")
        label_placeholder = tf.placeholder(tf.float32, [None, output_dim], name="label_placeholder")
        output, train_op, mse_loss = build_model(input_placeholder, label_placeholder, output_dim)

        sess.run(tf.global_variables_initializer())

        iterations = args.dagger_iterations if args.dagger else 1

        while iterations > 0:
            iterations -= 1
            if args.dagger:
                print("Dagger Iteration " + str(args.dagger_iterations - iterations))

            for i in range(args.epoch):
                training_loss = 0
                for batch_i in range(batch_count):
                    batch_start = batch_i * args.batch_size
                    inputs = X_train[batch_start : batch_start + args.batch_size]
                    labels = y_train[batch_start : batch_start + args.batch_size]
                    _, batch_loss = sess.run([train_op, mse_loss], feed_dict={
                        input_placeholder: inputs, label_placeholder: labels})
                    training_loss += batch_loss * len(inputs)
                test_loss = sess.run(mse_loss, feed_dict={
                    input_placeholder: X_test, label_placeholder: y_test})
                if i > args.epoch - 5:
                    print("Epoch " + str(i))
                    print("Training loss: ", training_loss / len(X_train))
                    print("Test loss: ", test_loss)

            if args.dagger:  # augment dataset with expert actions taken on imitation observations
                assert args.expert_policy_file is not None
                return_mean, return_std, (X_train_dagger, X_test_dagger, y_train_dagger, y_test_dagger)\
                     = prepare_dagger_data(input_placeholder, output)
                X_train = np.append(X_train, X_train_dagger, axis=0)
                X_test = np.append(X_test, X_test_dagger, axis=0)
                y_train = np.append(y_train, y_train_dagger, axis=0)
                y_test = np.append(y_test, y_test_dagger, axis=0)
                return_means.append(return_mean)
                return_stds.append(return_std)

        if args.dagger:
            print("return_means", return_means)
            print("return_stds", return_stds)

        # save model
        if args.save:
            dagger = "-dagger-%iiter-%idagger_rollouts" % (
                args.dagger_iterations, args.dagger_num_rollouts) if args.dagger else ""
            save_filename = args.training_data_file[:-4] + dagger + '-epoch%i.ckpt' % args.epoch
            save_path = os.path.join(OUTPUT_DIR , save_filename)
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            inputs = {"input_placeholder": input_placeholder}
            outputs = {"output": output}
            tf.saved_model.simple_save(sess, save_path, inputs, outputs)
            print("Model saved in %s" % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data_file', type=str)

    parser.add_argument("--dagger", action='store_true', help="run dagger algorithm")
    parser.add_argument("--expert_policy_file", type=str, help="required if running dagger")
    parser.add_argument("--dagger_iterations", type=int, default=30)
    parser.add_argument("--dagger_num_rollouts", type=int, default=100)

    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--regularization_weight", type=float, default=0.001)

    parser.add_argument("--save", action='store_true', help="save the trained tf model")
    args = parser.parse_args()

    args.envname = "-".join(args.training_data_file.split("-")[:2])

    run()