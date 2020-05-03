#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import datetime
from pathlib import Path
import continuous_mountain_car_evaluator


# This class is a bare version of tfp.distributions.Normal
class Normal:
    def __init__(self, loc, scale):
        self.loc = tf.convert_to_tensor(loc, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

    def log_prob(self, x):
        log_unnormalized = -0.5 * tf.math.squared_difference(x / self.scale, self.loc / self.scale)
        log_normalization = 0.5 * np.log(2. * np.pi) + tf.math.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * np.log(2. * np.pi) + tf.math.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * tf.ones_like(self.loc)

    def sample_n(self, n, seed=None):
        shape = tf.concat([[n], tf.broadcast_dynamic_shape(tf.shape(self.loc), tf.shape(self.scale))], axis=0)
        sampled = tf.random.normal(shape=shape, mean=0., stddev=1., dtype=tf.float32, seed=seed)
        return sampled * self.scale + self.loc


class Network:
    def __init__(self, env_, args_):
        assert len(env_.action_shape) == 1
        self.env = env_
        self.args = args_
        action_components = env_.action_shape[0]

        self.entropy_regularization = args.entropy_regularization

        # DONE: Create `model`, which: processes `states`. Because `states` are
        # vectors of tile indices, you need to convert them to one-hot-like
        # encoding. I.e., for batch example i, state should be a vector of
        # length `weights` with `tiles` ones on indices `states[i,
        # 0..`tiles`-1] and the rest being zeros.
        inputs = tf.keras.Input(shape=(env_.weights,))

        # The model computes `mus` and `sds`, each of shape [batch_size, action_components].
        # Compute each independently using `states` as input, adding a fully connected
        # layer with args.hidden_layer units and ReLU activation. Then:
        # - For `mus` add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required [-1,1] range, you can apply
        #   `tf.tanh` activation.
        mus_hidden = tf.keras.layers.Dense(args_.hidden_layer, 'relu')(inputs)
        mus = tf.keras.layers.Dense(action_components, 'tanh')(mus_hidden)

        # - For `sds` add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        sds_hidden = tf.keras.layers.Dense(args_.hidden_layer, 'relu')(inputs)
        sds = tf.keras.layers.Dense(action_components, tf.nn.softplus)(sds_hidden)

        # The model also computes `values`, starting with `states` and
        # - add a fully connected layer of size args.hidden_layer and ReLU activation
        # - add a fully connected layer with 1 output and no activation
        values_hidden = tf.keras.layers.Dense(args_.hidden_layer, 'relu')(inputs)
        values = tf.keras.layers.Dense(1)(values_hidden)

        self.model = tf.keras.Model(inputs=inputs, outputs=[mus, sds, values])
        self.optimizer = tf.optimizers.Adam(args_.learning_rate)

    @tf.function
    def _train(self, states_, actions_, returns_):
        with tf.GradientTape() as tape:
            # DONE: Run the model on given states and compute
            # `sds`, `mus` and `values`. Then create `action_distribution` using
            # `Normal` distribution class and computed `mus` and `sds`.
            mus, sds, values = self.model(states_, training=True)
            action_distribution = Normal(mus, sds)

            # DONE: Compute `loss` as a sum of three losses:
            # - negative log probability of the `actions` in the `action_distribution`
            #   (using `log_prob` method). You need to sum the log probabilities
            #   of subactions for a single batch example (using `tf.reduce_sum` with `axis=1`).
            #   Then weight the resulting vector by `(returns - tf.stop_gradient(values))`
            #   and compute its mean.
            loss_1 = tf.math.reduce_mean(
                (returns_ - tf.stop_gradient(values)) * tf.reduce_sum(-action_distribution.log_prob(actions_), axis=1)
            )
            # - negative value of the distribution entropy (use `entropy` method of
            #   the `action_distribution`) weighted by `args.entropy_regularization`.
            loss_2 = tf.math.reduce_mean(self.args.entropy_regularization * -action_distribution.entropy())
            # - mean square error of the `returns` and `values`.
            loss_3 = tf.math.reduce_mean(tf.losses.mean_squared_error(returns_, values))
            loss = loss_1 + loss_2 + loss_3
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def _encode_states(self, states_):
        return tf.reduce_sum(tf.one_hot(np.array(states_, np.int32), self.env.weights, axis=2), axis=1)

    def train(self, states_, actions_, returns_):
        states_ = self._encode_states(states_)
        actions_ = np.array(actions_, np.float32)
        returns_ = np.array(returns_, np.float32)
        self._train(states_, actions_, returns_)

    @tf.function
    def _predict(self, states_):
        return self.model(states_, training=False)

    def predict_actions(self, states_):
        states_ = self._encode_states(states_)
        mus, sds, _ = self._predict(states_)
        return mus.numpy(), sds.numpy()

    def predict_values(self, states_):
        _, _, values = self._predict(self._encode_states(states_))
        return values.numpy()[:, 0]


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--evaluate_each", default=128, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=48, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=32, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.000_05, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--tiles", default=32, type=int, help="Tiles to use.")
    parser.add_argument("--workers", default=64, type=int, help="Number of parallel workers.")
    parser.add_argument("--threshold", default=95, type=float, help="Threshold to pass in training.")
    parser.add_argument("--use_pretrained", action='store_true', default=True,
                        help="Whether or not to use pretrained networks.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
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
    env = continuous_mountain_car_evaluator.environment(tiles=args.tiles)
    action_lows, action_highs = env.action_ranges

    # Construct the network
    network = Network(env, args)

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)

    saved_model_path = Path(__file__).parent / 'paac_continuous_model_weights'
    training = not args.use_pretrained

    summary_writer = tf.summary.create_file_writer(
        str(Path(__file__).parent / 'paac-cont-logs' / f'train-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    )
    summary_writer.set_as_default()
    cycle = 0

    if training:
        import mlflow
        experiment_name = str(Path(__file__).stem)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_params(args.__dict__)
            while training:
                # Training
                for _ in range(args.evaluate_each):
                    # DONE: Choose actions using network.predict_actions.
                    # using np.random.normal to sample action and np.clip
                    # to clip it using action_lows and action_highs,
                    mus_, sds_ = network.predict_actions(states)
                    actions = np.clip(np.random.normal(mus_, sds_), action_lows, action_highs)

                    # DONE: Perform steps by env.parallel_step
                    step_list = env.parallel_step(actions)

                    # DONE: Compute return estimates by
                    # - extracting next_states from steps
                    next_states = [next_state for next_state, _, _, _ in step_list]
                    # - computing value function approximation in next_states
                    next_state_value = network.predict_values(next_states)
                    # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
                    dones = np.array([done for _, _, done, _ in step_list], dtype=np.int32)
                    rewards = np.array([reward for _, reward, _, _ in step_list])
                    returns = rewards + (1 - dones) * args.gamma * next_state_value

                    # DONE: Train network using current states, chosen actions and estimated returns
                    network.train(states, actions, returns)
                    states = next_states

                # Periodic evaluation
                returns = []
                for _ in range(args.evaluate_for):
                    returns.append(0)
                    # noinspection PyRedeclaration
                    state, done = env.reset(), False
                    while not done:
                        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                            env.render()

                        action = network.predict_actions([state])[0][0]
                        state, reward, done, _ = env.step(action)
                        returns[-1] += reward

                mean = np.mean(returns)
                print("Evaluation of {} episodes: {}".format(args.evaluate_for, mean))
                tf.summary.scalar('Mean reward', mean, step=cycle)
                mlflow.log_metric('Mean reward', mean, step=cycle)
                tf.summary.scalar('LR', network.optimizer.learning_rate.numpy(), step=cycle)
                mlflow.log_metric('LR', network.optimizer.learning_rate.numpy(), step=cycle)
                cycle += 1

                # Check for episode count threshold and required score threshold.
                if np.mean(returns) > args.threshold:
                    training = False
                    network.model.save_weights(str(saved_model_path / 'model'))

    if args.use_pretrained:
        import paac_continuous_model
        network.model.load_weights(str(saved_model_path / 'model'))

    # On the end perform final evaluations with `env.reset(True)`
    while True:
        # noinspection PyRedeclaration
        state, done = env.reset(True), False
        while not done:
            # DONE: Compute action `action_p` using `network.predict` and current `state`
            action = network.predict_actions([state])[0][0]
            state, _, done, _ = env.step(action)
