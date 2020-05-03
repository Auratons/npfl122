#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import datetime
from pathlib import Path
import gym_evaluator


class Network:
    def __init__(self, env_, args_):
        # DONE: Similarly to reinforce, define two models:
        # - _policy, which predicts distribution over the actions
        # - _value, which predicts the value function
        # Use independent networks for both of them, each with
        # `args.hidden_layer` neurons in one hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        self.policy = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(env_.state_shape),
                tf.keras.layers.Dense(args_.hidden_layer, activation="relu"),
                tf.keras.layers.Dense(env_.actions, activation='softmax')
            ],
            name='policy'
        )
        self.policy.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args_.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy()
        )
        self.value = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(env_.state_shape),
                tf.keras.layers.Dense(args_.hidden_layer, activation="relu"),
                tf.keras.layers.Dense(1)
            ],
            name='value'
        )
        self.value.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args_.learning_rate),
            loss='mse'
        )

    def train(self, states_, actions_, returns_):
        states_ = np.array(states_, np.float32)
        actions_ = np.array(actions_, np.int32)
        returns_ = np.array(returns_, np.float32)
        # DONE: Train the policy network using policy gradient theorem.
        values = self.value.predict_on_batch(states_)[:, 0]
        self.policy.train_on_batch(states_, actions_, sample_weight=(returns_ - values))
        # DONE: Train value network using MSE.
        self.value.train_on_batch(states_, returns_)

    def predict_actions(self, states_):
        states_ = np.array(states_, np.float32)
        return self.policy.predict_on_batch(states_).numpy()

    def predict_values(self, states_):
        states_ = np.array(states_, np.float32)
        return self.value.predict_on_batch(states_)[:, 0]


if __name__ == "__main__":
    # There is a bug in TF 2.0 which causes the `*_on_batch` methods not to use `tf.function`.
    assert [int(i) for i in tf.__version__.split('.')] >= [2, 1, 0]
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=128, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=64, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=32, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.000_05, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--workers", default=256, type=int, help="Number of parallel workers.")
    parser.add_argument("--threshold", default=460, type=float, help="Threshold to pass in training.")
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
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(env, args)

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)

    saved_model_path = Path(__file__).parent / 'paac_models_weights'
    training = not args.use_pretrained

    summary_writer = tf.summary.create_file_writer(
        str(Path(__file__).parent / 'logs' / f'train-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
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

                # Lower learning rate
                # if cycle == 178:
                #     network.policy.optimizer.learning_rate = args.learning_rate / 2
                #     network.value.optimizer.learning_rate = args.learning_rate / 2

                for _ in range(args.evaluate_each):
                    # DONE: Choose actions using network.predict_actions
                    actions = np.array(
                        [np.random.choice(env.actions, p=prob) for prob in network.predict_actions(states)]
                    )
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
                    returns = rewards + args.gamma * (1 - dones) * next_state_value

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

                        probabilities = network.predict_actions([state])[0]
                        action = np.argmax(probabilities)
                        state, reward, done, _ = env.step(action)
                        returns[-1] += reward

                mean = np.mean(returns)
                print("Evaluation of {} episodes: {}".format(args.evaluate_for, mean))
                tf.summary.scalar('Mean reward', mean, step=cycle)
                mlflow.log_metric('Mean reward', mean, step=cycle)
                tf.summary.scalar('LR', network.policy.optimizer.learning_rate.numpy(), step=cycle)
                mlflow.log_metric('LR', network.policy.optimizer.learning_rate.numpy(), step=cycle)
                cycle += 1

                # Check for episode count threshold and required score threshold.
                if np.mean(returns) > args.threshold:
                    training = False
                    network.policy.save_weights(str(saved_model_path / 'policy'))
                    network.value.save_weights(str(saved_model_path / 'value'))

    if args.use_pretrained:
        import paac_model
        network.policy.load_weights(str(saved_model_path / 'policy'))
        network.value.load_weights(str(saved_model_path / 'value'))

    # On the end perform final evaluations with `env.reset(True)`
    while True:
        # noinspection PyRedeclaration
        state, done = env.reset(True), False
        while not done:
            # DONE: Compute action `action_p` using `network.predict` and current `state`
            action_p = network.predict_actions(state[np.newaxis])[0]
            # Choose greedy action this time
            action = np.argmax(action_p)
            state, _, done, _ = env.step(action)
