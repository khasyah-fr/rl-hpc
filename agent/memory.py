import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.masks = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.masks),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def __len__(self):
        return len(self.states)

    def store_memory(self, state, mask, action, probs, vals, reward, done):
        self.states += [state.tolist()]
        self.masks += [mask.tolist()]
        self.actions += [action.tolist()]
        self.probs += [probs.item()]
        self.vals += [vals.item()]
        self.rewards += [reward]
        self.dones += [done]

    def clear_memory(self):
        self.states = []
        self.masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []