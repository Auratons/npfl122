#!/usr/bin/env python3
import argparse
import sys
import scipy.special
import numpy as np

class MultiArmedBandits():
    def __init__(self, bandits, episode_length, seed=42):
        self._generator = np.random.RandomState(seed)

        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(self._generator.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = self._generator.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}

parser = argparse.ArgumentParser()
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episodes", default=100, type=int, help="Training episodes.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
parser.add_argument("--c", default=1, type=float, help="Confidence level in ucb (if applicable).")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial value function levels (if applicable).")

def main(args):
    # Fix random seed
    np.random.seed(args.seed)

    # Create environment
    env = MultiArmedBandits(args.bandits, args.episode_length)
    
    all_episodes_mean = np.zeros(args.episodes, dtype=np.float64)
    # We use actions to be called 0 to NUMBER_OF_ACTIONS. (For that we use
    # np.random.choices and np.argmax everywhere.)
    NUMBER_OF_ACTIONS = args.bandits

    for episode in range(args.episodes):
        env.reset()

        one_episode_mean_reward = 0.0

        # DONE: Initialize parameters (depending on mode).
        if args.mode in ["greedy", "ucb"]:
            q_est = np.repeat(np.float64(args.initial), repeats=NUMBER_OF_ACTIONS)
            action_counts = np.zeros(NUMBER_OF_ACTIONS, dtype=np.int64)
        elif args.mode == "gradient":
            h = np.zeros(NUMBER_OF_ACTIONS, dtype=np.float64)

        done = False
        cycle = 0
        while not done:
            # DONE: Action selection according to mode
            if args.mode == "greedy":
                is_greedy = np.random.random_sample() < args.epsilon
                action = np.argmax(q_est) if not is_greedy else np.random.randint(NUMBER_OF_ACTIONS)
            elif args.mode == "ucb":
                # This is a very ugly, but transparent way how to solve zero counts resulting into
                # infinite values without any zero division warnings.
                counts = action_counts.copy()
                counts[action_counts == 0] = 1
                # q_est.copy() is used since we do not want to amend the q_est by addition,
                # cycle + 1 to solve t == 0.
                to_choose_from = q_est.copy() + args.c * np.sqrt(np.log(cycle + 1) / counts)
                to_choose_from[action_counts == 0] = np.inf
                action = np.argmax(to_choose_from)
            elif args.mode == "gradient":
                softmax = scipy.special.softmax(h)
                action = np.random.choice(NUMBER_OF_ACTIONS, p=softmax)

            _, reward, done, _ = env.step(action)

            # DONE: Update parameters
            if args.mode in ["greedy", "ucb"]:
                # https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides/?01#39 and
                # https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides/?01#47
                action_counts[action] += 1
                alpha = args.alpha if args.alpha != 0.0 else 1 / action_counts[action]
                q_est[action] = q_est[action] + alpha * (reward - q_est[action])
            elif args.mode == "gradient":
                # https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides/?01#60
                one_hot = np.zeros(NUMBER_OF_ACTIONS, dtype=np.int64)
                one_hot[action] = 1
                h += args.alpha * reward * (one_hot - softmax)
            
            # Iterative arithmetic mean calculation.
            one_episode_mean_reward = one_episode_mean_reward + (reward - one_episode_mean_reward) / cycle if cycle != 0 else reward
            cycle += 1

        all_episodes_mean[episode] = one_episode_mean_reward

    # DONE: For every episode, compute its average reward (a single number),
    # obtaining `args.episodes` values. Then return the final score as
    # mean and standard deviation of these `args.episodes` values.
    return np.mean(all_episodes_mean), np.std(all_episodes_mean)

if __name__ == "__main__":
    mean, std = main(parser.parse_args())
    # Print the mean and std for ReCodEx to validate
    print("{:.2f} {:.2f}".format(mean, std))
