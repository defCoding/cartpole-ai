import gym, math, numpy
from qtable import QTable

env = gym.make('CartPole-v0')

n_episodes = 100000
env._max_episode_steps = 10000

def find_nearest(ls, target):
    """
    Finds the value in a sorted list that is closest to the given number.

    Params:
        ls : [int] - the sorted list of numbers
        target : int - the number you are comparing to

    Returns:
        int - the value in the list closest to the target
    """
    # Cover edge cases first
    if target >= ls[-1]:
        return ls[-1]
    elif target <= ls[0]:
        return ls[0]
    
    def get_closer(a, b):
        diff_a = abs(target - a)
        diff_b = abs(target - b)

        return a if diff_a <= diff_b else b
   
    # Do binary search
    start = 0
    end = len(ls)

    while start < end:
        mid = (start + end) // 2
        
        if ls[mid] == target:
            return ls[mid]
        elif ls[mid] < target:
            # Check to see if ls[mid] or its right neighbor is closest.
            if mid < len(ls) - 1 and target < ls[mid + 1]:
                return get_closer(ls[mid], ls[mid + 1])

            start = mid
        else:
            # Check to see if ls[mid] or its left neighbor is closest.
            if mid > 0 and target > ls[mid - 1]:
                return get_closer(ls[mid], ls[mid - 1])

            end = mid


def discretize_observation_space(observation_space_bounds, steps):
    """
    Given the bounds of an observation space, return a set of discrete states for each
    factor of the observation space as the new discretized observation space.

    Params:
        observation_space_bounds : [(int, int)] - the bounds of the observation space
        steps : [int] - the interval between each discrete point for each factor

    Returns:
        [[int]] - a list of list of possible states for each factor in the observation space.
    """
    discrete_space = []

    for idx, bounds in enumerate(observation_space_bounds):
        discrete_states = []
        for i in numpy.arange(bounds[0], bounds[1], steps[idx]):
            discrete_states.append(i)

        discrete_space.append(discrete_states)

    return discrete_space

def discretize_state(state, observation_space):
    """
    Takes a state and finds its closest discretized state from the discretized observation
    space for that state.

    Params:
        state : (int) - the state to discretize
        observation_space : [[int]] - the discretized observation space, a list of list of
            possible values for each factor

    Returns:
        (int) - the discretized space
    """
    discretized_state = []

    for idx, factor in enumerate(state):
        discretized_state.append(find_nearest(observation_space[idx], factor))

    return tuple(discretized_state)

# CartPole environment describes observations as: # Observation:
#    Num Observation         Min                     Max
#    0   Cart Position       -4.8                    4.8
#    1   Cart Velocity       -Inf                    Inf
#    2   Pole Angle          -0.418 rad (-24 deg)    0.418 rad (24 deg)
#    3   Pole Velocity       -Inf                    Inf

bounds = list(zip(env.observation_space.low, env.observation_space.high))

# Velocity bounds by default are infinite, so rebind them.
bounds[1] = [-1, 1] 
bounds[3] = [-math.radians(50), math.radians(50)]

observation_space = discretize_observation_space(bounds, [15, 5, math.radians(1), math.radians(2)])
actions = [i for i in range(env.action_space.n)]
table = QTable(observation_space, actions)

prev_score = 0

for episode in range(n_episodes):
    observation = env.reset()
    state_action_pairs = []
    t_steps_taken = 0

    while True:
        env.render()
        state = discretize_state(observation, observation_space)
        action = table.decide_action(state)
        state_action_pairs.append((state, action))
        observation, reward, done, info = env.step(action)
        t_steps_taken += 1

        if done:
            break

    print(f'Finished episode {episode} at time {t_steps_taken}.')

    for t_step, sa_pair in enumerate(state_action_pairs):
        state, action = sa_pair
        next_state = state_action_pairs[t_step + 1][0] if t_step < len(state_action_pairs) - 1 else discretize_state(observation, observation_space)
        table.process_step(state, action, next_state, t_steps_taken - t_step)

    if episode > 3000 and t_steps_taken > prev_score:
        table.update_exploration_rate()

    prev_score = t_steps_taken
