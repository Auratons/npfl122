#!/usr/bin/env python3

'''
I can confirm that the algorithm converges faster than the one without the baseline.
But still, it takes 400 episodes. Spending time on gridsearch for shrinking the number
to 200 does not seem worth it to me, so I use the pretrained model trained with the
settings below.
'''

import numpy as np
import tensorflow as tf
from pathlib import Path
import cart_pole_evaluator
import reinforce_with_baseline_model


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
        with tf.GradientTape() as tape:
            neglogp = tf.keras.losses.sparse_categorical_crossentropy(y_true=actions, y_pred=self.call(states))
            outputs = tf.reduce_mean(neglogp * returns)
        gradients = tape.gradient(outputs, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def predict(self, states_):
        return self.call(states_)

    @tf.function
    def call(self, states_):
        net = self.input_layer(states_)
        for layer in self.hidden_layers:
            net = layer(net)
        action_preferences = self.output_layer(net)
        return action_preferences


class RFB:
    def __init__(self, env_, args_, name):
        # DONE: Define suitable model. Apart from the model defined in `reinforce`,
        # define also another model `baseline`, which produces one output
        # (using a dense layer without activation).
        #
        # Use Adam optimizer with given `args.learning_rate` for both models.
        self.reinforce = Network(env_, args_, name)
        self.baseline = tf.keras.models.Sequential([
            tf.keras.layers.Dense(args_.hidden_layer_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.baseline.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args_.learning_rate),
            loss='mse'
        )

    def train(self, states, actions, returns):
        # DONE: Train the model using the states, actions and observed returns.
        # You should:
        # - compute the predicted baseline using the `baseline` model
        # - train the policy model, using `returns - predicted_baseline` as weights
        #   in the sparse crossentropy loss
        # - train the `baseline` model to predict `returns`
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns, np.float32)
        predicted_baseline = self.baseline(states)
        self.baseline.train_on_batch(states, returns)
        self.reinforce.train(states, actions, returns - predicted_baseline)

    def predict(self, states_):
        # DONE: Predict distribution over actions for the given input states. Return
        # only the probabilities, not the baseline.
        return self.reinforce.predict(states_)


if __name__ == "__main__":
    # There is a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.
    assert [int(i) for i in tf.__version__.split('.')] >= [2, 1, 0]
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.004, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--threshold", default=496, type=float, help="Threshold to pass in training.")
    parser.add_argument("--use_pretrained", action='store_true', default=False, help="Whether or not to use pretrained policy.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
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
    network = RFB(env, args, 'reinforce_with_baseline')
    saved_model_path = f'{Path(__file__).parent}/reinforce_with_baseline_model_weights/'
    training = True if not args.use_pretrained else False

    while training:
        # Perform episode
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # DONE: Compute action probabilities using `network.predict` and current `state`
            action_p = network.predict(state[np.newaxis])[0].numpy()

            # DONE: Choose `action` according to `probabilities` distribution (np.random.choice can be used)
            action = np.random.choice(env.actions, p=action_p)

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # DONE: Compute returns by summing rewards (with discounting)
        returns = [0.0] * len(rewards)
        g = 0.0
        for r_idx in range(len(rewards) - 1, -1, -1):
            g = args.gamma * g + rewards[r_idx]
            returns[r_idx] = g

        # Train using the generated batch
        network.train(states, actions, returns)

        # Check for episode count threshold and required score threshold.
        if (args.episodes is not None and env.episode >= args.episodes) or (
            env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
            training = False
            network.reinforce.save_weights(saved_model_path)

    if args.use_pretrained:
        network.reinforce.load_weights(saved_model_path)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # DONE: Compute action `action_p` using `network.predict` and current `state`
            action_p = network.predict(state[np.newaxis])[0].numpy()
            # Choose greedy action this time
            action = np.argmax(action_p)
            state, reward, done, _ = env.step(action)
