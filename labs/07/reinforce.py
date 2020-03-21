#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from pathlib import Path
import cart_pole_evaluator
import reinforce_model


MAX_EPISODE_LENGTH = 500


# noinspection PyUnresolvedReferences
class Network(tf.keras.Model):
    def __init__(self, env_, args_, name):
        super(Network, self).__init__(name=name)
        self.input_layer = tf.keras.layers.Dense(args_.hidden_layer_size, activation="relu")
        self.hidden_layers = [
            tf.keras.layers.Dense(args_.hidden_layer_size, activation="relu") for _ in range(args_.hidden_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(env_.actions, activation=tf.keras.activations.softmax)

        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args_.learning_rate)
        )

    def train(self, states, actions, returns):
        # DONE: Train the model using the states, actions and observed returns.
        # Use `returns` as weights in the sparse crossentropy loss.
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        states, actions, returns = states.reshape((-1,) + tuple(env.state_shape)), actions.reshape((-1,) + tuple(env.action_shape)), returns.flatten()
        states, actions, returns = tf.convert_to_tensor(states), tf.convert_to_tensor(actions), tf.convert_to_tensor(returns)
        with tf.GradientTape() as tape:
            neglogp = tf.keras.losses.sparse_categorical_crossentropy(y_true=actions, y_pred=self.call(states))
            outputs = tf.reduce_mean(neglogp * returns)
        gradients = tape.gradient(outputs, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    @tf.function
    def predict(self, states_):
        return self.call(states_)

    @tf.function
    def call(self, states_):
        net = self.input_layer(states_)
        for layer in self.hidden_layers:
            net = layer(net)
        action_preferences = self.output_layer(net)
        return action_preferences


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=6, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--threshold", default=496, type=float, help="Threshold to pass in training.")
    parser.add_argument("--use_pretrained", action='store_true', help="Whether or not to use pretrained policy.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args, name='reinforce')
    saved_model_path = f'{Path(__file__).parent}/reinforce_model_weights/'
    training = True if not args.use_pretrained else False

    while training:
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states = np.zeros((MAX_EPISODE_LENGTH,) + tuple(env.state_shape))
            actions = np.zeros((MAX_EPISODE_LENGTH,) + tuple(env.action_shape))
            rewards = np.zeros((MAX_EPISODE_LENGTH,))
            state, done = env.reset(), False
            step = 0
            while not done and step < MAX_EPISODE_LENGTH:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # DONE: Compute action probabilities using `network.predict` and current `state`
                action_p = network.predict(state[np.newaxis])[0].numpy()

                # DONE: Choose `action` according to `probabilities` distribution (np.random.choice can be used)
                action = np.random.choice(env.actions, p=action_p)

                next_state, reward, done, _ = env.step(action)

                states[step] = state
                actions[step] = action
                rewards[step] = reward

                state = next_state
                
                step += 1

            # DONE: Compute returns by summing rewards (with discounting)
            returns = np.zeros_like(rewards)
            g = 0.0
            for r_idx in range(len(rewards) - 1, -1, -1):
                g = args.gamma * g + rewards[r_idx]
                returns[r_idx] = g

            # DONE: Add states, actions and returns to the training batch
            batch_states.append(states)
            batch_actions.append(actions)
            batch_returns.append(returns)

            if env.episode == 8970:
                network.save_weights(saved_model_path.rstrip('/') + '8970/')

        # Train using the generated batch
        network.train(batch_states, batch_actions, batch_returns)

        # Check for episode count threshold and required score threshold.
        if (args.episodes is not None and env.episode >= args.episodes) or (
            env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
            training = False
            network.save_weights(saved_model_path)

    if args.use_pretrained:
        network.load_weights(saved_model_path)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # DONE: Compute action `action_p` using `network.predict` and current `state`
            action_p = network.predict(state[np.newaxis])[0].numpy()
            # Choose greedy action this time
            action = np.argmax(action_p)
            state, reward, done, _ = env.step(action)
