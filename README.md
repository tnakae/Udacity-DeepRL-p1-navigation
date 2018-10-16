[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# About this project
**[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)**
is MOOCs lecture by [Udacity](https://www.udacity.com/) in which DeepRL experts
teach basic knowledge and implementation techniques about Deep RL via online video
(mainly Youtube contents), and there are a lot of implementation tutorials
in which each student has to implement various DeepRL algorithms.

This repository of DeepRL source code is work of 1st "Project" in this Nanodegree.
This is at the end of Lecture Part2, **Value-Based Methods** including
**Deep Q-Learning** and other developed methods (Double DQN, Prioritized Replay..)
The project goal is to implement smart agent walking around game simulator
which is created by using Unity ML-Agents([github repository](https://github.com/Unity-Technologies/ml-agents))

# Project Details
![Trained Agent][image1]

In the simulator, there are two types of bananas, yellow ones and blue ones.
If he catches yellow one, he gets positive reward (+1). But if he catches blue one,
he gets negative reward (-1). Thus, the goal of the agent is to collect
as many yellow bananas as possible while avoiding blue bananas.

Students need to implement DeepRL agent using Python and Pytorch.

The simulation contains a single agent that navigates a large environment.
At each time step, it has four actions at its disposal:

- `0` - walk forward
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space has `37` dimensions and contains the agent's velocity,
along with ray-based perception of objects around agent's forward direction.

The task is episodic, and in order to solve the environment, the agent must
get an average score of +13 over 100 consecutive episodes.

# Getting Started
1. Check [this nanodegree's prerequisite](https://github.com/udacity/deep-reinforcement-learning/#dependencies), and follow the instructions.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in `bin/` directory, and unzip (or decompress) the file.

# Instructions
To train the agent, start jupyter notebook, open `TrainDQN.ipynb`
and execute! For more information, please check instructions
inside the notebook.

# Additional informations
- [Performance Report](./Report.md) : Result report of training score
when using Double-DQN as agent.
