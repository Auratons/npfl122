#!/usr/bin/env python3
import numpy as np
import collections
import pathlib
import lunar_lander_av_func_embedded
import lunar_lander_evaluator


def greedy_policy(env, state, epsilon, action_value_function):
    if epsilon is not None and np.random.random_sample() < epsilon:
        return np.random.choice(env.actions)
    else:
        return np.argmax(action_value_function[state])  # Greedy action


class Episode:
    def __init__(self):
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        is_done = False
        self.states = collections.deque(maxlen=(n + 1))
        self.actions = collections.deque(maxlen=(n + 1))
        self.rewards = collections.deque(maxlen=(n + 1))
        self.timestep = 0
        self.tau = 0
        self.end_timestep = np.iinfo(np.int32).max
        return is_done


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=6000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threshold", default=100, type=float, help="Threshold to pass in training.")
    parser.add_argument("--algorithm", default='q-learn', type=str,
                        help="Either 'ne-sarsa', 'off-mc', 'on-mc' or 'q-learn'.")
    parser.add_argument("--use_pretrained", action='store_true', default=False,
                        help="Whether or not to use pretrained policy.")

    parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    parser.add_argument("--n", "-n", default=3, type=int, help="N parameter for the n-step expected Sarsa algorithm.")
    args = parser.parse_args()

    # Shortcuts
    epsilon = args.epsilon
    alpha = args.alpha
    gamma = args.gamma
    n = args.n

    # Create the environment
    env = lunar_lander_evaluator.environment()

    action_value_function = np.zeros((env.states, env.actions), dtype=np.float32)
    training = not args.use_pretrained

    if args.algorithm == 'ne-sarsa':
        # Implementation of n-step expected Sarsa follows.
        # (Baseline algorithm from Sutton's book p. 147.)
        epi = Episode()
        # Perform training episodes of n-step Sarsa
        while training:
            state_0, done = env.reset(), epi.reset()
            action_0 = greedy_policy(env, state_0, epsilon, action_value_function)
            epi.states.append(state_0)
            epi.actions.append(action_0)
            epi.rewards.append(0)  # To make natural indexing to work (for tau==0 R_{t+1} is on index 1).

            while epi.tau != epi.end_timestep - 1:
                if epi.timestep < epi.end_timestep:
                    state_t_plus_1, reward_t_plus_1, done, _ = env.step(epi.actions[epi.timestep % (n + 1)])  # action_t
                    epi.rewards.append(reward_t_plus_1)
                    epi.states.append(state_t_plus_1)
                    if done:
                        epi.end_timestep = epi.timestep + 1
                    else:
                        action_t_plus_1 = greedy_policy(env, state_t_plus_1, epsilon, action_value_function)
                        epi.actions.append(action_t_plus_1)

                epi.tau = epi.timestep - n + 1

                if epi.tau >= 0:
                    return_estimate = sum(
                        np.power(gamma, i - epi.tau - 1) * epi.rewards[i % (n + 1)]
                        # Plus one to include the bound in the sum.
                        for i in range(epi.tau + 1, min(epi.tau + n, epi.end_timestep) + 1)
                    )
                    if epi.tau + n < epi.end_timestep:
                        action_distribution = action_value_function[epi.states[(epi.tau + n) % (n + 1)]]
                        # Expected Sarsa (all q_a have prob eps / |actions| except for the greedy
                        # action that has 1 - eps + eps / |actions|).
                        return_estimate += (
                            np.power(gamma, n) * (
                                # Compute expected value for epsilon-soft policy (epsilon-greedy precisely)
                                sum(q_a * (epsilon / env.actions) for q_a in action_distribution)
                                + (1 - epsilon) * np.argmax(action_distribution)
                            )
                        )
                    q_tau = action_value_function[epi.states[epi.tau % (n + 1)], epi.actions[epi.tau % (n + 1)]].copy()    
                    action_value_function[epi.states[epi.tau % (n + 1)], epi.actions[epi.tau % (n + 1)]] = \
                        q_tau + alpha * (return_estimate - q_tau)

                epi.timestep += 1

            # Check for episode count threshold and required score threshold.
            # noinspection PyProtectedMember,DuplicatedCode
            if (args.episodes is not None and env.episode >= args.episodes) or (
                    env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
                training = False

            if args.epsilon_final is not None:
                # Exponential decay.
                epsilon = np.exp(
                    np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)])
                )

            # noinspection DuplicatedCode
            if args.alpha_final is not None:
                # Exponential decay.
                alpha = np.exp(
                    np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])
                )

            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
    elif args.algorithm in ['off-mc', 'on-mc']:
        # Sutton's book, p. 111
        if args.algorithm == 'off-mc':
            counts = np.zeros((env.states, env.actions), dtype=np.int32)
            policy = np.zeros((env.states,), dtype=np.int32)

            def behavior_policy(_):
                return np.random.choice(env.actions)
        else:
            returns = collections.defaultdict(list)
            policy = np.tile(np.float32(1 / env.actions), (env.states, env.actions))

            def behavior_policy(state_):
                return np.random.choice(env.actions, p=policy[state_])

        while training:
            episode = []
            state = env.reset()
            done = False
            runs = 0

            # Generate an episode following behavior on or off policy
            while not done:
                action = behavior_policy(state)
                next_state, reward, done, _ = env.step(action)
                episode.append((state, action, reward))
                state = next_state
                runs += 1

            g = 0.0
            if args.algorithm == 'off-mc':
                w = 1.0
                for t in range(runs - 2, 0, -1):
                    state_t, action_t, reward_t_plus_1 = episode[t + 1]
                    g = args.gamma * g + reward_t_plus_1
                    counts[state_t, action_t] += w
                    old_q = action_value_function[state_t, action_t].copy()
                    action_value_function[state_t, action_t] = old_q + (w / counts[state_t, action_t]) * (
                        g - old_q
                    )
                    policy[state_t] = np.argmax(action_value_function[state_t])
                    if action_t != policy[state_t]:
                        break
                    w *= env.actions  # We have uniformly random behavior policy.
            else:
                # noinspection DuplicatedCode
                for t in range(runs - 2, 0, -1):
                    state_t, action_t, reward_t_plus_1 = episode[t + 1]
                    g = args.gamma * g + reward_t_plus_1
                    returns[(state_t, action_t)].append(g)
                    action_value_function[state_t, action_t] = np.mean(returns[(state_t, action_t)])
                    starred_action = np.argmax(action_value_function[state_t])
                    for a in range(env.actions):
                        if a == starred_action:
                            policy[state_t, a] = 1 - epsilon + epsilon / env.actions
                        else:
                            policy[state_t, a] = epsilon / env.actions

            # Check for episode count threshold and required score threshold.
            # noinspection PyProtectedMember
            if (args.episodes is not None and env.episode >= args.episodes) or (
                    env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
                training = False

            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
    elif args.algorithm == 'q-learn':
        episode = 0
        while training:
            # Perform a training episode
            state, done = env.reset(), False

            if episode > 50 and episode % 2 == 1:
                while not done:
                    # For the exploratory part we conduct epsilon greedy policy for getting the action.
                    action = greedy_policy(env, state, epsilon, action_value_function)
                    next_state, reward, done, _ = env.step(action)
                    action_value_function[state, action] +=  alpha * (
                        reward + \
                        args.gamma * np.max(action_value_function[next_state]) - \
                        action_value_function[state, action]
                    )
                    state = next_state
            else:
                state, trajectory = env.expert_trajectory()
                for (action, reward, next_state) in trajectory:
                    action_value_function[state, action] +=  alpha * (
                        reward + \
                        args.gamma * np.max(action_value_function[next_state]) - \
                        action_value_function[state, action]
                    )
                    state = next_state

            episode += 1

            # Check for episode count threshold and required score threshold.
            if (args.episodes is not None and episode >= args.episodes) or (
                episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
                training = False

            if args.epsilon_final is not None:
                # Exponential decay.
                epsilon = np.exp(
                    np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)])
                )

            if args.alpha_final is not None:
                # Exponential decay.
                alpha = np.exp(
                    np.interp(episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])
                )

            if args.render_each and episode and episode % args.render_each == 0:
                env.render()

    filename = str(pathlib.Path(__file__).parent / 'lunar-lander-av-func.npz')
    if not args.use_pretrained:
        np.savez_compressed(filename, policy=action_value_function)
    else:
        try:
            action_value_function = np.load(filename)['policy']
        except FileNotFoundError:
            action_value_function = np.load('lunar-lander-av-func.npz')['policy']

    # Perform last 100 evaluation episodes
    for _ in range(100):
        # noinspection PyRedeclaration
        state = env.reset(start_evaluate=True)
        done = False
        while not done:
            action = greedy_policy(env, state, None, action_value_function)
            next_state, reward, done, _ = env.step(action)
            state = next_state

        if args.render_each and env.episode and env.episode % args.render_each == 0:
            env.render()
