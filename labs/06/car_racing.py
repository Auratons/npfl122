#!/usr/bin/env python3
import datetime
import collections
import random
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


class ReplayBuffer(collections.deque):
    def __init__(self, maxlen):
        super(ReplayBuffer, self).__init__(maxlen=maxlen)

    # noinspection PyMethodOverriding
    def append(self, state_, action_, reward_, done_, next_state_):
        super(ReplayBuffer, self).append(Transition(state_, action_, reward_, done_, next_state_))

    def sample(self, sample_size):
        batch = random.sample(self, sample_size)
        states = np.array([transition.state for transition in batch], dtype=np.float32)
        actions = np.array([transition.action for transition in batch], dtype=np.int32)
        rewards = np.array([transition.reward for transition in batch], dtype=np.float32)
        dones = np.array([transition.done for transition in batch], dtype=np.int32)
        next_states = np.array([transition.next_state for transition in batch], dtype=np.float32)
        return states, actions, rewards, dones, next_states


class DQN(tf.keras.Model):
    def __init__(self, name):
        # There is a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.
        assert [int(i) for i in tf.__version__.split('.')] >= [2, 1, 0]
        super(DQN, self).__init__(name=name)
        self.layer_C1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')
        self.layer_C2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')
        # self.layer_C3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')
        self.layer_F1 = tf.keras.layers.Flatten()
        self.layer_D1 = tf.keras.layers.Dense(256, activation='relu')
        self.layer_Out = tf.keras.layers.Dense(len(DISCRETE_ACTIONS), activation='linear')

    @tf.function
    def call(self, states):
        net = self.layer_C1(states)
        net = self.layer_C2(net)
        # net = self.layer_C3(net)
        net = self.layer_F1(net)
        net = self.layer_D1(net)
        output_q_values = self.layer_Out(net)
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
        self.build(
            input_shape=(
                None,
                env_.state_shape[0] - MARGIN,
                env_.state_shape[1],
                args_.frame_history
            )
        )

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args_.learning_rate),
            loss=tf.keras.losses.MeanSquaredError()
        )

    def build(self, input_shape):
        self.target_network.build(input_shape)
        self.target_network.trainable = False
        return super(DDQN, self).build(input_shape)

    def train_on_batch(self, *args_, **kwargs_):
        self._step_for_target_network_update += 1
        if self._step_for_target_network_update % self._target_freq == 0:
            self._step_for_target_network_update = 0
            for main, target in zip(self.trainable_variables, self.target_network.trainable_variables):
                target.assign(main)
        return super(DDQN, self).train_on_batch(*args_, **kwargs_)

    def train(self, states, actions, rewards, dones, next_states, sample_weight=None, *_):
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
        batch_size = len(actions)

        q_values_t = self.predict(states)
        q_values_t_plus_1 = self.predict(next_states)
        argmax_actions = tf.math.argmax(q_values_t_plus_1, axis=1, output_type=tf.int32)

        q_updates = rewards + (1 - dones) * args.gamma * \
            tf.gather_nd(
                self.target_network.predict(next_states), tf.stack((tf.range(batch_size), argmax_actions), axis=1)
            )
        # Numpy: self.predict(next_states)[(tf.range(len(argmax_actions), dtype=tf.int32), argmax_actions)]
        q_values = q_values_t.numpy()
        q_values[(np.arange(batch_size), actions)] = q_updates
        loss_ = self.train_on_batch(states, q_values, sample_weight)
        return loss_


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=4, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=4, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--use_pretrained", action='store_true', default=False, help="Use the pretrained policy.")

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--target_freq", default=100, type=int, help="Target network update frequency in train steps.")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate.")

    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
    parser.add_argument("--threshold", default=235, type=float, help="Threshold to pass in training.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
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

    # Create the environment
    env = car_racing_evaluator.environment(args.frame_skip)

    # Construct the network
    network = DDQN(env, args)
    print(network.summary())

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = ReplayBuffer(1_000_000)
    epsilon = args.epsilon
    saved_model_path = f'{Path(__file__).parent}/q_network_model_weights/'
    training = not args.use_pretrained

    summary_writer = tf.summary.create_file_writer(
        str(Path(__file__).parent / 'logs' / f'train-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    )
    summary_writer.set_as_default()
    avg_episode_td_error = tf.keras.metrics.Mean(name='TD_error', dtype=tf.float32)

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

        # Record the learning rate and epsilon for the current episode.
        tf.summary.scalar('learning_rate', args.learning_rate, step=env.episode)
        tf.summary.scalar('epsilon', epsilon, step=env.episode)

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
            replay_buffer.append(state, action, reward, done, next_states_4th_slice)

            if len(replay_buffer) >= args.batch_size:
                loss = network.train(*replay_buffer.sample(args.batch_size))
                avg_episode_td_error.update_state(loss)

            state = np.concatenate((state[..., 1:], next_states_4th_slice[..., np.newaxis]), axis=-1)

        # Record the average TD error for training steps performed in the current episode and reset it.
        tf.summary.scalar('TD_error', avg_episode_td_error.result(), step=env.episode)
        avg_episode_td_error.reset_states()
        # noinspection PyProtectedMember
        moving_expected_return = np.mean(env._episode_returns[-100:])
        tf.summary.scalar('100_epoch_return_mean', moving_expected_return, step=env.episode)

        if (args.episodes is not None and env.episode >= args.episodes) or \
           (args.threshold is not None and moving_expected_return > args.threshold):
            training = False
            network.save_weights(saved_model_path)

        if args.epsilon_final is not None:
            epsilon = np.exp(
                np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)])
            )

    if args.use_pretrained:
        network.load_weights(saved_model_path)

    # After training (or loading the model), you should run the evaluation:
    while True:
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = np.argmax(network.predict(state[np.newaxis])[0])
            state, reward, done, _ = env.step(DISCRETE_ACTIONS[action])
