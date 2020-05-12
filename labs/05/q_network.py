#!/usr/bin/env python3
import collections
import tensorflow as tf
import cart_pole_evaluator
import numpy as np
import random
import datetime
import embedded_data
from pathlib import Path


class Network(tf.keras.Model):
    def __init__(self, env, args):
        # DONE: Create a suitable network

        # Warning: If you plan to use Keras `.train_on_batch` and/or `.predict_on_batch`
        # methods, pass `experimental_run_tf_function=False` to compile. There is
        # a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.

        # Otherwise, if you are training manually, using `tf.function` is a good idea
        # to get good performance.
        super(Network, self).__init__(name='DQN')
        self.input_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")
        self.hidden_layers = [tf.keras.layers.Dense(args.hidden_layer_size, activation="relu") for _ in range(args.hidden_layers)]
        self.output_layer = tf.keras.layers.Dense(env.actions, activation="linear")
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
        # Define our metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'{Path(__file__).parent}/logs/train' + current_time
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    @tf.function
    def train(self, states, q_values):
        with tf.GradientTape() as tape:
            outputs = self(states)
            loss = self.loss(q_values, outputs)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)


    @tf.function
    def call(self, states):
        net = self.input_layer(states)
        for layer in self.hidden_layers:
            net = layer(net)
        output_q_values = self.output_layer(net)
        return output_q_values

    @tf.function
    def predict(self, states):
        return self.call(states)


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=2000, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=0, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--threshold", default=420, type=float, help="Threshold to pass in training.")
    parser.add_argument("--use_pretrained", action='store_true', default=False, help="Whether or not to use pretrained policy.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
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
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=1000000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])
    epsilon = args.epsilon
    saved_model_path = f'{Path(__file__).parent}/q_network_model_weights/'
    training = True if not args.use_pretrained else False

    while training:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # DONE: compute action using epsilon-greedy policy. You can compute
            # the q_values of a given state using
            #   q_values = network.predict(np.array([state], np.float32))[0]
            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = np.argmax(network.predict(state.reshape((1, -1)).astype(np.float32))[0])

            next_state, reward, done, _ = env.step(action)

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # DONE: If the replay_buffer is large enough, preform a training batch
            # of `args.batch_size` uniformly randomly chosen transitions.
            #
            # After you choose `states` and suitable targets, you can train the network as
            #   network.train(states, ...)
            if len(replay_buffer) >= args.batch_size:
                batch = random.sample(replay_buffer, args.batch_size)
                states = np.array([transition.state for transition in batch], dtype=np.float32)
                actions = np.array([transition.action for transition in batch], dtype=np.int32)
                next_states = np.array([transition.next_state for transition in batch], dtype=np.float32)
                rewards = np.array([transition.reward for transition in batch], dtype=np.float32)
                dones = np.array([transition.done for transition in batch], dtype=np.int32)

                q_values = network.predict(states).numpy()
                q_updates = rewards + (1 - dones) * (args.gamma * np.max(network.predict(next_states).numpy(), axis=1))
                q_values[(np.arange(len(actions)), actions)] = q_updates

                network.train(states, q_values)

            state = next_state

        with network.train_summary_writer.as_default():
            tf.summary.scalar('mse_loss', network.train_loss.result(), step=env.episode)

        template = 'Episode {}, MSE loss: {:.1f}'
        if env.episode % 10 == 0:
            print(template.format(env.episode, network.train_loss.result()))
        network.train_loss.reset_states()

        # Check for episode count threshold and required score threshold.
        if (args.episodes is not None and env.episode >= args.episodes) or (
            env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
            training = False
            network.save_weights(saved_model_path)

        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))

    if args.use_pretrained:
        network.load_weights(saved_model_path)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(network.predict(state.reshape((1, -1)).astype(np.float32))[0])
            state, reward, done, _ = env.step(action)
