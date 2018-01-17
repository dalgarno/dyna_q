import random


class Agent(object):
    def __init__(self, gridworld, eps=0.1, alpha=0.1, gamma=0.95, n=0):
        super(Agent, self).__init__()
        self.gridworld = gridworld
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.states = self.gridworld.available_states()
        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }  # up, down, left, right
        self.state_actions = [(x, y) for x in self.states for y in self.actions]
        self.action_values = self.init_action_values()
        self.model_values = self.init_model_values()
        self.current_state = self.gridworld.start_state
        self.visited_states = set()
        self.visited_states.add(self.current_state)
        self.visited_states_actions = self.init_actions_taken_in_states()

    def init_action_values(self):
        return {k: 0 for k in self.state_actions}

    def init_model_values(self):
        return {k: (None, 0) for k in self.state_actions}

    def init_actions_taken_in_states(self):
        return {k: [] for k in self.states}

    def choose_action(self, state):
        best_val = float('-inf')
        best_actions = []
        for action in self.actions:
            val = self.action_values[(state, action)]
            if val > best_val:
                best_val = val
                best_actions = [action]
            elif val == best_val:
                best_actions.append(action)

        # take a random action
        if random.random() < self.eps:
            return random.choice(list(self.actions.keys()))
        return random.choice(best_actions)

    def reset(self):
        self.action_values = self.init_action_values()
        self.model_values = self.init_model_values()
        self.current_state = self.gridworld.start_state
        self.visited_states = set()
        self.visited_states.add(self.current_state)
        self.visited_states_actions = self.init_actions_taken_in_states()

    def take_action(self, action):
        new_state = tuple(map(
            lambda x, y: x + y,
            self.current_state,
            self.actions[action])
        )

        if new_state == self.gridworld.goal_state:
            return 1, new_state

        if new_state in self.states:
            return 0, new_state

        return 0, self.current_state

    def max_state_action_val(self, state):
        max_val = float('-inf')
        for action in self.actions:
            if self.action_values[(state, action)] > max_val:
                max_val = self.action_values[(state, action)]
        return max_val

    def update_state_action(self, s, a, r, s_prime):
        val_s_a = self.action_values[(s, a)]

        self.action_values[(s, a)] = val_s_a + self.alpha * (
                r + self.gamma * self.max_state_action_val(s_prime) - val_s_a
        )

    def run_episode(self):
        action_list = []
        total_reward = 0
        state_action_list = []
        while True:
            action = self.choose_action(self.current_state)
            reward, new_state = self.take_action(action)

            action_list.append(action)
            total_reward += reward
            state_action_list.append((new_state, action))

            self.visited_states_actions[self.current_state].append(action)

            self.update_state_action(self.current_state, action, reward, new_state)
            self.model_values[(self.current_state, action)] = reward, new_state

            if new_state == self.gridworld.goal_state:
                self.current_state = self.gridworld.start_state
                return action_list, total_reward, state_action_list

            for planning_step in range(self.n):
                prev_seen_state = random.sample(self.visited_states, 1)[0]
                rand_action = random.choice(self.visited_states_actions[prev_seen_state])
                reward, s_prime = self.model_values[(prev_seen_state, rand_action)]
                self.update_state_action(prev_seen_state, rand_action, reward, s_prime)

            self.visited_states.add(new_state)
            self.current_state = new_state
