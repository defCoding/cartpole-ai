# cartpole-ai
Using Q-Learning to make an AI that can balance a pole on a cart ([OpenAI Gym's Cartpole](https://gym.openai.com/envs/CartPole-v0/)).

## Context
The only "real" AI I've done so far is the NEAT Algorithm, and I wanted to try my hand at other learning algorithms. I'm particularly interested in Deep Q-Learning, but before tackling that, I wanted to learn Q-Learning first.

## How it Works
### Learning
| Q-Learning Parameter | Value |
| --- | --- |
| Learning Rate | 0.05 |
| Discount Rate | 0.99 |
| Exploration Decay | 0.01 |

The AI runs for 3000 episodes with an exploration rate of 1 to allow it to become familiar with the environment. After the 3000 episodes, the exploration rate can decay if the AI performs better in the current episode compared to the previous episode. This allows the AI to maintain a high exploration rate so that optimal actions can be found.

### Rewards
The Q-Table is not updated iteratively over the course of an episode - instead, each action that is taken is recorded along with the state at which the action is taken at. After the episode ends, the state-action pair is assigned a reward equal to the difference between the total time steps and the time step at which the action-pair occurred.

### Improvements
The Cartpole environments provides 4 parameters for each state observation:

- Cart Position
- Cart Velocity
- Pole Angle
- Pole Angular Velocity

Each of these values are continuous on a certain bound, which is not compatible with the Q-Tables of Q-Learning. As such, these values needed to be discretized onto some set of discrete values per parameter.

Improving on the set of discrete values (for example, for pole angle, allowing for smaller steps closer to the 0 degrees from the vertical and larger steps when the pole is more tilted could allow for finer state-action pairings) should increase performance.

## How to Use
Install the `gym` library first:

```
$ pip install gym
```

Then run the `cartpole.py` file:

```
$ python cartpole.py
```
