#!/usr/bin/env python3
import numpy as np
import collections
import embedded_data
import cart_pole_evaluator


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2000, type=int, help="Max number of training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threshold", default=499.5, type=int, help="Threshold to pass in training.")
    parser.add_argument("--use_pretrained", action='store_true', help="Whether or not to use pretrained policy.")

    parser.add_argument("--epsilon", default=0.18, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0025, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # DONE: Implement Monte-Carlo RL algorithm.
    # More precisely: On-policy every-visit MC control (for epsilon-soft policies), p. 101 from Sutton's book.

    action_value_function = np.zeros((env.states, env.actions), dtype=np.float32)
    returns = collections.defaultdict(list)
    policy = np.tile(np.float32(1 / env.actions), (env.states, env.actions))
    
    training = True
    epsilon = args.epsilon

    while training and not args.use_pretrained:
        episode = []
        state = env.reset()
        done = False
        runs = 0

        # Generate an episode following policy
        while not done:
            action = np.random.choice(env.actions, p=policy[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            runs += 1
        
        g = 0
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
        if (args.episodes is not None and env.episode >= args.episodes) or (
            env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
            training = False

        if args.epsilon_final is not None:
            # Exponential decay.
            epsilon = np.exp(
                np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)])
            )

        if args.render_each and env.episode and env.episode % args.render_each == 0:
            env.render()
        
    if args.use_pretrained:
        # In ReCodEx, there is 3 minute limit on computation, so we pretrained and embedded
        # obtained policy with ../embed.py script.
        policy = np.load('pretrained_policy.npz')['policy']

    # Perform last 100 evaluation episodes
    for _ in range(100):
        state = env.reset(start_evaluate=True)
        done = False
        while not done:
            action = np.random.choice(env.actions, p=policy[state])
            next_state, reward, done, _ = env.step(action)
            state = next_state
