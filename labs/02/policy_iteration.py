#!/usr/bin/env python3
import numpy as np

class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 3 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states = 11

    actions = ["↑", "→", "↓", "←"]

    @staticmethod
    def step(state, action):
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability, state, action):
        if state >= 5: state += 1
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not(new_x >= 4 or new_x < 0  or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        if state >= 5: state -= 1
        return [probability, +1 if state == 3 else -100 if state == 6 else 0, state]

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=4, type=int, help="Number of policy evaluation/improvements to perform.")
    parser.add_argument("--iterations", default=4, type=int, help="Number of iterations in policy evaluation step.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
    args = parser.parse_args()

    # DONE: Implement policy iteration algorithm, with `args.steps` steps of
    # policy evaluation/policy improvement. During policy evaluation, use the
    # current value function and perform `args.iterations` applications of the
    # Bellman equation. Perform the policy evaluation synchronously (i.e., do
    # not overwrite the current value function when computing its improvement,
    # only overwrite the previous value function after each iteration).

    def bellman(state_to_recompute, action, curr_value_function):
        new_value = 0.0
        possible_outcomes = GridWorld.step(state_to_recompute, action)
        for probability, reward, new_state in possible_outcomes:
            new_value += probability * (reward + args.gamma * curr_value_function[new_state])
        return new_value

    def policy_evaluation(curr_policy, value_function):
        value_function = value_function.copy()
        for _ in range(0, args.iterations):
            v_func = np.zeros(GridWorld.states, dtype=np.float32)
            for state in range(0, GridWorld.states):
                v_func[state] = bellman(state, curr_policy[state], value_function)
            value_function = v_func
        return value_function

    # This is a very simplified version of algorithm from slides
    # (https://ufal.mff.cuni.cz/~straka/courses/npfl122/1920/slides/?02#67).

    # 1. Initialization
    # Start with zero value function and "go North" policy (0th in the list)
    value_function = np.zeros(GridWorld.states, dtype=np.float32)
    policy = np.zeros(GridWorld.states, dtype=np.int32)

    for _ in range(0, args.steps):
        # 2. Policy evaluation phase
        value_function = policy_evaluation(policy, value_function)

        # 3. Policy improvement
        for state in range(0, GridWorld.states):
            policy[state] = np.argmax(
                [bellman(state, a, value_function) for a, _ in enumerate(GridWorld.actions)]
            )

    # DONE: The final greedy policy is in `policy`

    # Print results
    for l in range(3):
        for c in range(4):
            state = l * 4 + c
            if state >= 5: state -= 1
            print("        " if l == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if l == 1 and c == 1 else GridWorld.actions[policy[state]], end="")
        print()
