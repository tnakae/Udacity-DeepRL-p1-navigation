# Summary
In this report, I describe training result using **Double DQN**.

# Learning Algorithm
Source code of learning algorithm is placed in `dqn/` directory.
The algorithm is composed mainly of next three parts.

## 1. Double DQN
As reinforcement learning algorithm, I implemented **Double DQN** (**DDQN**).
Because Double DQN overcome the tendency which DQN overestimates
the reward of action, DDQN can achieve better performance then DQN on average
(but I don't have enough time to check it..)

Inside DDQN, I used two layer neural network as Q-Value Estimator.
Hidden layers are composed of ``State -> 64 -> LeakyReLU -> 64 -> LinearReLU -> Action``

## 2. Experience Replay
Similarly to original DQN paper, I implemented **Experience Replay**.
In this technique, D-DQN model is trained by mini-batch from replay buffer.

## 3. Epsilon Greedy
Agent select next action based on Epsilon Greedy. At probability epsilon,
agent select at random from action space.
The value of epsilon is set 0.999, and decrease gradually with time until 0.001.

# Rewards Result
This is plot of rewards when training.
At Episode 524, agent performance met the criteria and stopped training.
(mean scores of last 100 episodes is above +13)

![plot of rewards](./Report_score.jpg)

# Ideas for Future Work
For further performance improvement, I need to implement next algorithm.
- Implement [Prioritized Experienced Replay](https://arxiv.org/abs/1511.05952)
- Grid search of hyper-parameter, especially Q-value model.

# Trained model
[Trained model (DDQN)](./checkpoint.pth)
