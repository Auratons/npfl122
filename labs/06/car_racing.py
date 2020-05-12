#!/usr/bin/env python3
import datetime
import collections
import random
import argparse
import mlflow
import mlflow.tensorflow
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage import color

import car_racing_evaluator

DISCRETE_ACTIONS = [
    # steer in range [-1, 1]
    # gas in range [0, 1]
    # brake in range [0, 1]
    [-1, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0.5, 0],
    [0, 0.5, 0],
    [1, 0.5, 0],
    # [-1, 0, 1],
    # [0, 0, 1],
    # [1, 0, 1],
]

# Strip bottom line of the screen with controls.
# Easier for the network to train based only on a state of sequence of images.
MARGIN = 11


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])


parser = argparse.ArgumentParser()
parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
parser.add_argument("--frame_skip", default=4, type=int, help="Repeat actions for given number of frames.")
parser.add_argument("--frame_history", default=4, type=int, help="Number of past frames to stack together.")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--use_pretrained", action='store_true', default=False, help="Use the pretrained policy.")

parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--target_freq", default=100, type=int, help="Target network update frequency in train steps.")
parser.add_argument("--learning_rate", default=0.00025, type=float, help="Learning rate.")
parser.add_argument("--memory_size", default=100_000, type=int, help="The size of prioritized replay buffer.")
parser.add_argument("--priority_alpha", default=0.6, type=float, help="Prioritisation factor: 0. uniform, 1. full.")
parser.add_argument("--priority_beta", default=1.0, type=float, help="Importance sampling: 0. none, 1. full.")
parser.add_argument("--priority_beta_final", default=None, type=float, help="Importance sampling: 0. none, 1. full.")
parser.add_argument("--priority_minimal", default=0.01, type=float, help="Minimal priority always present.")

parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
parser.add_argument("--threshold", default=250, type=float, help="Threshold to pass in training.")


def setup(args_):
    # Fix random seeds and number of threads
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args_.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args_.threads)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    mlflow.tensorflow.autolog()
    mlflow.log_params(args_.__dict__)
    mlflow.set_tag('DISCRETE_ACTION', str(DISCRETE_ACTIONS))


def log_metric(key, value, step):
    tf.summary.scalar(key, value, step=step)
    mlflow.log_metric(key, float(value), step=step)


class PrioritizedReplayBuffer:
    def __init__(self, env_, args_):
        slice_shape = (env_.state_shape[0] - MARGIN, env_.state_shape[1])
        state_shape = slice_shape + (args_.frame_history,)
        memory_shape = (args_.memory_size,)

        self.state      = np.zeros(memory_shape + state_shape, dtype=np.float32)
        self.action     = np.zeros(memory_shape,               dtype=np.int32)
        self.reward     = np.zeros(memory_shape,               dtype=np.float32)
        self.done       = np.zeros(memory_shape,               dtype=np.int32)
        self.next_state = np.zeros(memory_shape + slice_shape, dtype=np.float32)
        self.priorities = np.zeros(memory_shape,               dtype=np.float32)
        self.curr_write_idx = 0
        self.curr_size = 0
        self.args = args_

    def __len__(self):
        return self.curr_size

    def append(self, state_, action_, reward_, done_, next_state_, priority):
        self.state[self.curr_write_idx] = state_
        self.action[self.curr_write_idx] = action_
        self.reward[self.curr_write_idx] = reward_
        self.done[self.curr_write_idx] = done_
        self.next_state[self.curr_write_idx] = next_state_
        self.priorities[self.curr_write_idx] = priority

        self.curr_write_idx += 1
        if self.curr_write_idx >= self.args.memory_size:
            self.curr_write_idx = 0

        if self.curr_size < self.args.memory_size:
            self.curr_size += 1

    def update(self, idx_, val):
        self.priorities[idx_] = val

    def sample(self):
        probabilities = self.priorities[:self.curr_size].copy() / np.sum(self.priorities[:self.curr_size])
        sample_idxs_ = np.random.choice(
            np.arange(self.curr_size),
            self.args.batch_size,
            p=probabilities
        )
        is_weights = np.power(self.curr_size * self.priorities[sample_idxs_].copy(), -self.args.priority_beta)
        is_weights = is_weights / np.max(is_weights)
        return (
            self.state[sample_idxs_],
            self.action[sample_idxs_],
            self.reward[sample_idxs_],
            self.done[sample_idxs_],
            self.next_state[sample_idxs_],
            is_weights,
            sample_idxs_
        )


class DQN(tf.keras.Model):
    def __init__(self, name):
        # There is a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.
        # noinspection PyUnresolvedReferences
        assert [int(i) for i in tf.__version__.split('.')] >= [2, 1, 0]
        super(DQN, self).__init__(name=name)
        layer_pool = []

        def wrap(layer, **kwargs):
            layer_pool.append({})
            for k_ in kwargs:
                layer_pool[-1][(repr(layer), k_)] = str(kwargs[k_])
            return layer(**kwargs)

        self._all_layers = [
            wrap(tf.keras.layers.Conv2D, filters=16, kernel_size=8, strides=2, activation='relu', padding='same'),
            wrap(tf.keras.layers.MaxPool2D, pool_size=2, strides=2, padding='same'),
            wrap(tf.keras.layers.Conv2D, filters=32, kernel_size=4, strides=2, activation='relu', padding='same'),
            wrap(tf.keras.layers.MaxPool2D, pool_size=2, strides=2, padding='same'),
            wrap(tf.keras.layers.Flatten),
            wrap(tf.keras.layers.Dense, units=64, activation='relu'),
            wrap(tf.keras.layers.Dense, units=len(DISCRETE_ACTIONS), activation='linear'),
        ]

        for idx_, d_ in enumerate(layer_pool):
            for (rep, k) in d_:
                mlflow.set_tag(rep.split('\'')[1] + f'_{idx_}_' + k, layer_pool[idx_][(rep, k)])

    @tf.function
    def call(self, states):
        assert len(self._all_layers) >= 2
        features = self._all_layers[0](states)
        for layer in self._all_layers[1:-1]:
            features = layer(features)
        output_q_values = self._all_layers[-1](features)
        return output_q_values

    @tf.function
    def predict(self, states):
        return self.call(states)


class DDQN(DQN):
    def __init__(self, env_, args_):
        super(DDQN, self).__init__('main_DQN')
        self.target_network = DQN('target_DQN')
        self._step_for_target_network_update = 0
        self._target_freq = args_.target_freq
        self.priority_alpha = args_.priority_alpha
        self.build(
            input_shape=(
                None,
                env_.state_shape[0] - MARGIN,
                env_.state_shape[1],
                args_.frame_history
            )
        )

        # self._huber_delta = 100
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args_.learning_rate),
            loss=tf.keras.losses.MeanSquaredError()  # Huber(self._huber_delta)
        )

    def build(self, input_shape):
        self.target_network.build(input_shape)
        self.target_network.trainable = False
        return super(DDQN, self).build(input_shape)

    def train_on_batch(self, *args_, **kwargs_):
        self._step_for_target_network_update += 1
        if self._step_for_target_network_update % self._target_freq == 0:
            self._step_for_target_network_update = 0
            for main, target in zip(self.variables, self.target_network.variables):
                target.assign(main)
        return super(DDQN, self).train_on_batch(*args_, **kwargs_)

    @tf.function
    def compute_q_values_updates(self, states, actions, rewards, dones, next_states):
        # Construct proper next_states as args.next_states volume by stacking
        # args.frame_history - 1 most recent frames from states + frame represented by old next_states.
        next_states = tf.concat(
                (
                    states[..., 1:],
                    # Equivalent to next_states[..., np.newaxis], from [None, X, Y] to [None, X, Y, 1].
                    tf.expand_dims(next_states, axis=-1)
                ),
                axis=-1
        )
        q_values_t = self.predict(states)
        q_values_t_plus_1 = self.predict(next_states)
        argmax_actions = tf.math.argmax(q_values_t_plus_1, axis=1, output_type=tf.int32)

        q_updates = rewards + args.gamma * tf.cast(1 - dones, dtype=tf.float32) * tf.gather_nd(
                self.target_network.predict(next_states), tf.stack((tf.range(actions.shape[0]), argmax_actions), axis=1)
            )
        # np: self.target_network.predict(next_states)[(tf.range(len(argmax_actions), dtype=tf.int32), argmax_actions)]

        return q_values_t, q_updates

    def compute_td_errors(self, q_updates, q_values_t):
        # return tf.keras.losses.Huber(self._huber_delta, tf.keras.losses.Reduction.NONE)(q_updates, q_values_t)
        # return tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.NONE)(q_updates, q_values_t)
        return tf.math.pow(tf.math.abs(q_updates - q_values_t), self.priority_alpha)

    def train(self, states, actions, rewards, dones, next_states, sample_weight):
        q_values_t, q_updates = self.compute_q_values_updates(states, actions, rewards, dones, next_states)
        q_values = q_values_t.numpy()
        q_values[(np.arange(len(q_values_t)), actions)] = q_updates

        td_errors_ = self.compute_td_errors(
            q_updates,
            tf.gather_nd(q_values_t, tf.stack((tf.range(actions.shape[0]), actions), axis=1))
        ).numpy()

        loss_ = self.train_on_batch(states, q_values, sample_weight)
        return loss_, td_errors_


if __name__ == "__main__":
    with mlflow.start_run():
        # Parse arguments
        args = parser.parse_args()
        setup(args)
        # Create the environment
        env = car_racing_evaluator.environment(args.frame_skip)
        # Construct the network
        network = DDQN(env, args)
        print(network.summary())

        saved_model_path = f'{Path(__file__).parent}/q_network_model_weights/'
        summary_writer = tf.summary.create_file_writer(
            str(Path(__file__).parent / 'logs' / f'train-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        )
        summary_writer.set_as_default()

        replay_buffer = PrioritizedReplayBuffer(env, args)
        epsilon = args.epsilon
        training = not args.use_pretrained
        max_curr_priority = args.priority_minimal
        avg_episode_td_error = tf.keras.metrics.Mean(name='TD_error', dtype=tf.float32)
        first_beta = args.priority_beta

        while training:
            if args.render_each and (env.episode + 1) % args.render_each == 0:
                env.render()
            # Gather episode's initial state.
            state, done = color.rgb2gray(env.reset())[:-MARGIN, ..., np.newaxis], False
            # Create initial state of shape (state.shape[0] - MARGIN, state.shape[1], frame_history)
            # with initial (state.shape[0] - MARGIN, state.shape[1], frame_history-1) zeros and the
            # last slice as game's grayscale initial frame.
            initial_zeros = np.zeros((state.shape[0], state.shape[1], args.frame_history - 1), dtype=state.dtype)
            state = np.concatenate((initial_zeros, state), axis=-1)

            # Perform the episode.
            while not done:
                # Compute action using epsilon-greedy policy.
                if np.random.rand() < epsilon:
                    action = np.random.choice(len(DISCRETE_ACTIONS))
                else:
                    action = np.argmax(network.predict(state[np.newaxis])[0])

                next_states_4th_slice, reward, done, _ = env.step(DISCRETE_ACTIONS[action])
                next_states_4th_slice = color.rgb2gray(next_states_4th_slice[:-MARGIN, ...])

                # Append state, action, reward, done and next_state to replay_buffer
                replay_buffer.append(
                    state, action, reward, done, next_states_4th_slice, max_curr_priority
                )
                if len(replay_buffer) >= args.batch_size:
                    s, a, r, d, n, sample_is_weights, sample_idxs = replay_buffer.sample()
                    loss, td_errors = network.train(s, a, r, d, n, sample_is_weights)
                    avg_episode_td_error.update_state(loss)
                    for idx, sample_idx in enumerate(sample_idxs):
                        replay_buffer.update(sample_idx, td_errors[idx])
                    max_curr_priority = np.max(td_errors)

                state = np.concatenate((state[..., 1:], next_states_4th_slice[..., np.newaxis]), axis=-1)

            # Record the average TD error for training steps performed in the current episode and reset it.
            log_metric('TD_error', avg_episode_td_error.result(), step=env.episode)
            avg_episode_td_error.reset_states()
            # noinspection PyProtectedMember
            moving_expected_return = np.mean(env._episode_returns[-100:])
            log_metric('100_epoch_return_mean', moving_expected_return, step=env.episode)
            log_metric('learning_rate', args.learning_rate, step=env.episode)
            log_metric('epsilon', epsilon, step=env.episode)
            log_metric('IS beta', args.priority_beta, step=env.episode)

            if (args.episodes is not None and env.episode >= args.episodes) or \
               (args.threshold is not None and moving_expected_return > args.threshold):
                training = False
                network.save_weights(saved_model_path)
                mlflow.log_artifact(saved_model_path)

            if args.epsilon_final is not None:
                epsilon = np.exp(
                    np.interp(
                        env.episode + 1, [0, 4 * args.episodes / 5], [np.log(args.epsilon), np.log(args.epsilon_final)]
                    )
                )
            if args.priority_beta_final is not None:
                args.priority_beta = np.interp(
                    env.episode + 1, [0, 4 * args.episodes / 5], [first_beta, args.priority_beta_final]
                )

        if args.use_pretrained:
            import car_racing_model
            network.load_weights(saved_model_path)

        # After training (or loading the model), you should run the evaluation:
        while True:
            # noinspection PyRedeclaration
            state, done = color.rgb2gray(env.reset(start_evaluate=True))[:-MARGIN, ..., np.newaxis], False
            initial_zeros = np.zeros((state.shape[0], state.shape[1], args.frame_history - 1), dtype=state.dtype)
            state = np.concatenate((initial_zeros, state), axis=-1)
            while not done:
                action = np.argmax(network.predict(state[np.newaxis])[0])
                # noinspection PyTypeChecker
                next_states_4th_slice, reward, done, _ = env.step(DISCRETE_ACTIONS[action])
                next_states_4th_slice = color.rgb2gray(next_states_4th_slice[:-MARGIN, ...])
                state = np.concatenate((state[..., 1:], next_states_4th_slice[..., np.newaxis]), axis=-1)
