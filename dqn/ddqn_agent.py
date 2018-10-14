import torch
from dqn import DQNAgent


class DDQNAgent(DQNAgent):
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Calculate target value
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_dash_local = self.qnetwork_local(next_states)
            Q_dash_target = self.qnetwork_target(next_states)
            argmax_action = torch.max(Q_dash_local, dim=1, keepdim=True)[1]
            Q_dash_max = Q_dash_target.gather(1, argmax_action)
            y = rewards + gamma * Q_dash_max * (1 - dones)
        self.qnetwork_target.train()

        # Predict Q-value
        self.optimizer.zero_grad()
        Q = self.qnetwork_local(states)
        y_pred = Q.gather(1, actions)

        # TD-error
        loss = torch.sum((y - y_pred)**2)

        # Optimize
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
