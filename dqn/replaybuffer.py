import numpy as np
import random
from collections import namedtuple, deque
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): target device name ("cuda:0" or "cpu")
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        experience_fields = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=experience_fields)
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        experiences = [e for e in experiences if e is not None]

        def to_torch(x):
            return torch.from_numpy(np.vstack(x))
        def to_torch_uint8(x):
            return torch.from_numpy(np.vstack(x).astype(np.uint8))

        states = to_torch([e.state for e in experiences]).float()
        actions = to_torch([e.action for e in experiences]).long()
        rewards = to_torch([e.reward for e in experiences]).float()
        next_states = to_torch([e.next_state for e in experiences]).float()
        dones = to_torch_uint8([e.done for e in experiences]).float()

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
