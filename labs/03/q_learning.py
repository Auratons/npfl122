#!/usr/bin/env python3
import numpy as np
import mountain_car_evaluator
import embedded_data
from pathlib import Path

def action_given_policy(state, params):
    """
    Epsilon greedy policy.

    Function's parameters are more general so that performing the last 100
    episodes can be used across more scripts in the same way.

    Params:
      state: State from which the chosen action will be taken.
      params: A cidctionary containing action-value function under key "avf",
              float epsilon under "eps" key and environments under "env" key.
    """
    epsilon = params['eps']
    action_value_function = params['avf']
    environment = params['env']

    if epsilon is not None and np.random.random_sample() < epsilon:
        return np.random.choice(environment.actions)  # Greedy action
    else:
        return np.argmax(action_value_function[state])


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2500, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--threshold", default=-130, type=float, help="Threshold to pass in training.")

    parser.add_argument("--alpha", default=0.6, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.6, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0000001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--use_pretrained", action='store_true', default=True, help="Whether or not to use pretrained action value function.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()

    # TODO: Implement Q-learning RL algorithm.
    #
    # The overall structure of the code follows.
    action_value_function = np.zeros((env.states, env.actions), dtype=np.float32)
    epsilon = args.epsilon
    alpha = args.alpha

    saved_action_value_path = f'{Path(__file__).parent}/q_learning_avf.npz'

    training = True if not args.use_pretrained else False
    while training:
        # Perform a training episode
        state, done = env.reset(), False

        while not done:
            # For the exploratory part we conduct epsilon greedy policy for getting the action.
            policy_parameters = {'env': env, 'avf': action_value_function, 'eps': epsilon}
            action = action_given_policy(state, policy_parameters)
            next_state, reward, done, _ = env.step(action)
            action_value_function[state, action] +=  alpha * (
                reward + \
                args.gamma * np.max(action_value_function[next_state]) - \
                action_value_function[state, action]
            )
            #print(reward)
            state = next_state

        # Check for episode count threshold and required score threshold.
        if (args.episodes is not None and env.episode >= args.episodes) or (
            env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
            training = False
            np.savez_compressed(saved_action_value_path, avf=action_value_function)

        if args.epsilon_final is not None:
            # Exponential decay.
            epsilon = np.exp(
                np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)])
            )

        if args.alpha_final is not None:
            # Exponential decay.
            alpha = np.exp(
                np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])
            )

        if args.render_each and env.episode and env.episode % args.render_each == 0:
            env.render()
        
    if args.use_pretrained:
        # In ReCodEx, there is 3 minute limit on computation, so we pretrained and embedded
        # obtained policy with ../embed.py script.
        action_value_function = np.load(saved_action_value_path)['avf']

    # For evaluation, we follow only best actions without exploring via epsilon greedy actions
    policy_parameters = {'env': env, 'avf': action_value_function, 'eps': None}

    # Perform last 100 evaluation episodes
    for _ in range(100):
        state = env.reset(start_evaluate=True)
        done = False
        while not done:
            action = action_given_policy(state, policy_parameters)
            next_state, reward, done, _ = env.step(action)
            state = next_state
