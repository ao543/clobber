import numpy as np


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])


    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards)

def load_experience(h5file):
    return ExperienceBuffer(states=np.array(h5file['experience']['states']),
                            actions=np.array(h5file['experience']['actions']),
                            rewards = np.array(h5file['experience']['rewards'])
                            )

class ExperienceBuffer:

    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def serialize(self, h5file):

        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)




class ExperienceCollector:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_actions = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []

    def record_decision(self, state, action):
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)

    def complete_episode(self, reward):
        #Test
        #print("hello")
        #print(self.states[1].shape)

        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        self.rewards += [reward for i in range(num_states)]
        self.current_episode_states = []
        self.current_episode_actions = []

    def to_buffer(self):
        return ExperienceBuffer(states = np.array(self.states),
                                actions = np.array(self.actions),
                                reward = np.array(self.rewards))

    def set_collector(self, collector):
        self.collector = collector




