#!/usr/bin/env python3
import collections
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
# noinspection PyUnresolvedReferences
import gym_evaluator


class Network:
    def __init__(self, env_, args_):
        # There is a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.
        # noinspection PyUnresolvedReferences
        assert [int(i) for i in tf.__version__.split('.')[:2]] >= [2, 1]
        assert len(env_.action_shape) == 1
        action_components = env_.action_shape[0]
        action_lows_, action_highs_ = map(tf.convert_to_tensor, env_.action_ranges)

        def rescale(sigmoid_output):
            return action_lows_ + (action_highs_ - action_lows_) * sigmoid_output

        input_states = tf.keras.Input(shape=tuple(env_.state_shape))
        input_actions = tf.keras.Input(shape=tuple(env_.action_shape))
        # DONE: Create `actor` network, starting with `input_states` and returning
        # `action_components` values for each batch example. Usually, one
        # or two hidden layers are employed. Each `action_component[i]` should
        # be mapped to range `[action_lows_[i]..action_highs_[i]]`, for example
        # using `tf.nn.sigmoid` and suitable rescaling.
        #
        # Then, create a target actor as a copy of the model using
        # `tf.keras.models.clone_model`.
        actor_hidden = tf.keras.layers.Dense(args_.hidden_layer, 'relu')(input_states)
        actor = rescale(tf.keras.layers.Dense(action_components, 'sigmoid')(actor_hidden))

        self.actor = tf.keras.Model(inputs=input_states, outputs=actor)
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args_.learning_rate),
            loss='mse'
        )

        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_actor.trainable = False

        # DONE: Create `critic` network, starting with `input_states` and `input_actions`
        # and producing a vector of predicted returns. Usually, `input_states` are fed
        # through a hidden layer first, and then concatenated with `input_actions` and fed
        # through two more hidden layers, before computing the returns.
        #
        # Then, create a target critic as a copy of the model using `tf.keras.models.clone_model`.
        critic_hidden = tf.keras.layers.Dense(args_.hidden_layer, 'relu')(input_states)
        critic_hidden = tf.keras.layers.concatenate([critic_hidden, input_actions])
        critic_hidden = tf.keras.layers.Dense(args_.hidden_layer, 'relu')(critic_hidden)
        critic_hidden = tf.keras.layers.Dense(args_.hidden_layer, 'relu')(critic_hidden)
        critic = tf.keras.layers.Dense(1)(critic_hidden)

        self.critic = tf.keras.Model(inputs=[input_states, input_actions], outputs=critic)
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args_.learning_rate),
            loss='mse'
        )

        self.target_critic = tf.keras.models.clone_model(self.critic)
        self.target_critic.trainable = False

        self.target_nets_update_freq = args_.targets_update_frequency
        self.target_update_count = 0
        self.target_tau = args_.target_tau

    @tf.function
    def _train(self, states_, actions_, returns_):
        # DONE: Train separately the actor and critic.
        #
        # Furthermore, update the weights of the target actor and critic networks
        # by using args.target_tau option.
        with tf.GradientTape() as tape:
            _actions = self.actor(states_, training=True)
            _values = self.critic.predict_on_batch([states_, _actions])
            loss = -tf.math.reduce_mean(_values)
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        self.critic.train_on_batch([states_, actions_], returns_)

        self.target_update_count += 1
        if self.target_update_count % self.target_nets_update_freq == 0:
            self.target_update_count = 0
            for main, target in zip(self.actor.variables, self.target_actor.variables):
                target.assign(self.target_tau * main + (1 - self.target_tau) * target)
            for main, target in zip(self.critic.variables, self.target_critic.variables):
                target.assign(self.target_tau * main + (1 - self.target_tau) * target)

    def train(self, states_, actions_, returns_):
        self._train(np.array(states_, np.float32), np.array(actions_, np.float32), np.array(returns_, np.float32))

    @tf.function
    def _predict_actions(self, states_):
        # DONE: Compute actions by the actor
        return self.actor.predict_on_batch(states_)

    def predict_actions(self, states_):
        return self._predict_actions(np.array(states_, np.float32)).numpy()

    @tf.function
    def _predict_values(self, states_):
        # DONE: Predict actions by the target actor and evaluate them using
        # target_critic.
        return self.target_critic.predict_on_batch([states_, self.target_actor.predict_on_batch(states_)])

    def predict_values(self, states_):
        return self._predict_values(np.array(states_, np.float32)).numpy()[:, 0]


class OrnsteinUhlenbeckNoise:
    # Ornstein-Uhlenbeck process.

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        self.state = np.zeros(shape, np.float32)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
    parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=32, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.000_5, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--target_tau", default=0.001, type=float, help="Target network update weight.")
    parser.add_argument("--targets_update_frequency", default=1, type=int, help="Frequency of target nets' uspdates.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--threshold", default=-180, type=float, help="Threshold to pass in training.")
    parser.add_argument("--use_pretrained", action='store_true', default=False,
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
    env = gym_evaluator.GymEnvironment(args.env)
    action_lows, action_highs = map(np.array, env.action_ranges)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    noise = OrnsteinUhlenbeckNoise(env.action_shape[0], 0., args.noise_theta, args.noise_sigma)

    saved_model_path = Path(__file__).parent / f'{Path(__file__).stem}_models_weights'
    training = not args.use_pretrained

    pth = Path(__file__)
    summary_writer = tf.summary.create_file_writer(
        str(pth.parent / f'logs-{pth.stem}' / f'train-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
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
                    # noinspection PyRedeclaration
                    state, done = env.reset(), False
                    noise.reset()
                    while not done:
                        # DONE: Perform an action and store the transition in the replay buffer
                        action = np.clip(
                            network.predict_actions([state])[0] + noise.sample(), action_lows, action_highs
                        )
                        next_state, reward, done, _ = env.step(action)
                        replay_buffer.append(Transition(state, action, reward, done, next_state))

                        # If the replay_buffer is large enough, perform training
                        if len(replay_buffer) >= args.batch_size:
                            batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                            states, actions, rewards, dones, next_states = zip(*[replay_buffer[i] for i in batch])
                            # DONE: Perform the training
                            next_state_value = network.predict_values(next_states)
                            returns = np.array(rewards, np.float32) + \
                                (1 - np.array(dones, np.float32)) * \
                                args.gamma * \
                                np.array(next_state_value, np.float32)
                            network.train(states, actions, returns)

                        state = next_state

                # Periodic evaluation
                returns = []
                for _ in range(args.evaluate_for):
                    returns.append(0)
                    # noinspection PyRedeclaration
                    state, done = env.reset(), False
                    while not done:
                        if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                            env.render()

                        action = network.predict_actions([state])[0]
                        state, reward, done, _ = env.step(action)
                        returns[-1] += reward

                mean = np.mean(returns)
                print("Evaluation of {} episodes: {}".format(args.evaluate_for, mean))
                tf.summary.scalar('Mean reward', mean, step=cycle)
                mlflow.log_metric('Mean reward', mean, step=cycle)
                tf.summary.scalar('LR', network.actor.optimizer.learning_rate.numpy(), step=cycle)
                mlflow.log_metric('LR', network.critic.optimizer.learning_rate.numpy(), step=cycle)
                cycle += 1

                # Check for episode count threshold and required score threshold.
                if np.mean(returns) > args.threshold:
                    training = False
                    network.actor.save_weights(str(saved_model_path / 'policy'))

    if args.use_pretrained:
        # noinspection PyUnresolvedReferences
        import ddpg_model
        network.actor.load_weights(str(saved_model_path / 'policy'))

    # On the end perform final evaluations with `env.reset(True)`
    while True:
        # noinspection PyRedeclaration
        state, done = env.reset(True), False
        while not done:
            # DONE: Compute action `action_p` using `network.predict` and current `state`
            action = network.predict_actions([state])[0]
            state, _, done, _ = env.step(action)
