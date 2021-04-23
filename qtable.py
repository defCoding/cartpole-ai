import numpy
import random

class QTable:
    """
    Note:
        A State type is a tuple containing the value of each factor for a given state
        A Factor is the value of a factor of a state.
        An Action type is an object denoting an action to be taken at states

    __init__:
        observation_space : [[float]] - a list of list of possible values for each factor of a state
        actions: [Action] - a list of possible actions
        learning_rate : float (optional) - the learning rate used in the Bellman equation (defaults to .05)
        discount_rate : float (optional) - the discount rate used in the Bellman equation (defaults to .99)
        exploration_decay : float (optional) - the rate at which the exploration rate decays after each episode (defaults to .01)
        
    Attributes:
        _table : ndarray(float) - an n-dimensional array containing the Q-Values stored in the table
        _states_idxs : [{Factor : int}] - a list of dictionaries that map a factor to its corresponding index in the Q-Table,
            the first dictionary corresponds to the first factor, the second dictionary corresponds to the second factor, etc.
        observation_space : [[float]] - see above in __init__()
        actions : [Action] - see above in __init__()
        learning_rate : float - see above in __init__()
        discount_rate : float - see above in __init__()
        exploration_decay : float - see above in __init__()
        explorate_rate : float - the rate at which the AI chooses to explore vs exploit
    """

    def __init__(self, observation_space, actions, learning_rate=0.05, discount_rate=0.99, exploration_decay=0.01):
        self._table = numpy.zeros(tuple([len(discrete_states) for discrete_states in observation_space]) + (len(actions),))
        self._states_idxs = [{state : idx for idx, state in enumerate(discrete_states)} for discrete_states in observation_space]
        self._actions_idxs = {action : idx for idx, action in enumerate(actions)}
        self.observation_space = observation_space
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_decay = exploration_decay
        self.exploration_rate = 1

    def _get_state_values(self, state):
        """
        Returns the list of Q-Values for actions at a given state.

        Params:
            state : State - the state of all factors.

        Returns:
            [float] - The list of Q-values for actions at that state
        """
        dim = self._table
        for idx, single_state in enumerate(state):
            dim = dim[self._states_idxs[idx][single_state]]

        return dim

    def _get_qvalue(self, state, action):
        """
        Gets the Q-Value stored in the Q-Table for the given state-action pair.

        Params:
            state : State - the state of all factors
            action : Action - the action at the given state

        Returns:
            float - the Q-Value for that state-action pair
        """
        action_qvalues = self._get_state_values(state)
        a_idx = self._actions_idxs[action]

        return action_qvalues[a_idx]

    def _update_qvalue(self, state, action, new_value):
        """
        Updates the Q-Value in the Q-Table for the given state-action pair.

        Params:
            state : State - the state of all factors
            action : Action - the action at the given state
        """
        action_qvalues = self._get_state_values(state)
        a_idx = self._actions_idxs[action]

        action_qvalues[a_idx] = new_value

    def _get_max_action(self, state):
        """
        Returns a tuple containing the optimal action at a given state and the Q-Value associated
        with that state-action pair.

        Params:
            state : State - the state containg the state of all factors to get the optimal action of

        Returns:
            (State, float) - tuple containing the state and its associated q-value
        """
        action_qvalues = self._get_state_values(state)
        max_idx = action_qvalues.argmax(axis = 0)

        return (self.actions[max_idx], action_qvalues[max_idx])

    def process_step(self, state, action, new_state, reward):
        """
        Updates the Q-Table with the reward and new state after taking the given action at the
        given state.

        Params:
            state : State - the state that the action is being taken at
            action : Action - the action being taken
            new_state : State - the next state after the action is taken
            reward : float - the reward of taking the action
        """
        old_qvalue = self._get_qvalue(state, action)
        new_qvalue = reward + self.discount_rate * self._get_max_action(new_state)[1]

        self._update_qvalue(state, action, (1 - self.learning_rate) * old_qvalue + self.learning_rate * new_qvalue)

    def decide_action(self, state):
        """
        Decides the next action given the current state.

        Params:
            state : State - the current state

        Returns:
            Action - the decided upon action
        """
        # Decide between exploration and exploitation
        if random.random() <= self.exploration_rate:
            return random.choice(self.actions) # Exploration
        else:
            return self._get_max_action(state)[0] # Exploitation

    def update_exploration_rate(self):
        """
        Updates the exploration rate with the decay.
        """
        self.exploration_rate *= 1 - self.exploration_decay
