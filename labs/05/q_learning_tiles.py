#!/usr/bin/env python3
import numpy as np
import mountain_car_evaluator
import pretrained_tile_weights
from pathlib import Path

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=3000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")
    parser.add_argument("--threshold", default=-104, type=float, help="Threshold to pass in training.")

    parser.add_argument("--alpha", default=0.7, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.001, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.7, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0000001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    parser.add_argument("--use_pretrained", action='store_true', default=True, help="Whether or not to use pretrained weights.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.weights, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    saved_weights_path = f'{Path(__file__).parent}/q_learning_tiles_weights.npz'

    evaluating = False if not args.use_pretrained else True
    while not evaluating:
        # Perform a training episode
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # DONE: Choose `action` according to epsilon-greedy strategy
            one_hot_of_state = np.zeros(env.weights, dtype=np.float64)
            one_hot_of_state[state] = 1.0
            qav_for_state = one_hot_of_state @ W

            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = np.argmax(np.sum(W[state], axis=0))

            next_state, reward, done, _ = env.step(action)

            one_hot_for_next_state = np.zeros(env.weights, dtype=np.float64)
            one_hot_for_next_state[next_state] = 1.0
            qav_for_next_state = one_hot_for_next_state @ W

            # DONE: Update W values
            W[:, action] += one_hot_of_state * alpha * (
                reward + \
                args.gamma * np.max(qav_for_next_state) - \
                qav_for_state[action]
            )
            state = next_state

            if done:
                break


        # DONE: Decide if we want to start evaluating
        if (args.episodes is not None and env.episode >= args.episodes) or (
            env.episode >= 100 and np.mean(env._episode_returns[-100:]) > args.threshold):
            evaluating = True
            np.savez_compressed(saved_weights_path, weights=W)

        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            if args.alpha_final:
                alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles
        
    if args.use_pretrained:
        # In ReCodEx, there is 3 minute limit on computation, so we pretrained and embedded
        # obtained policy with ../embed.py script.
        W = np.load(saved_weights_path)['weights']

    # Perform the final evaluation episodes
    while True:
        state, done = env.reset(evaluating), False
        while not done:
            # DONE: choose action as a greedy action
            one_hot_of_state = np.zeros(env.weights, dtype=np.float32)
            one_hot_of_state[state] = 1
            qav_for_state = one_hot_of_state @ W
            action = np.argmax(qav_for_state)
            state, reward, done, _ = env.step(action)
