
# Project II: Continuous Control

## Introduction

The project uses Deep Reinforcement Learning algorithm to train an agent to navigate and collect fruits in a large, square world.

![Trained Agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

## Project Details

#### State-Space

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Rewards and Completion

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The second version of the problem which is solved in this project takes into account the presence of many agents. In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically:
-   After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
-   This yields an  **average score**  for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.
## Getting Started & Instructions

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
 
**_Version 2: Twenty (20) Agents_**

-   Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
-   Mac OSX:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
-   Windows (32-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
-   Windows (64-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
2. Place the file in the in the root folder, and unzip (or decompress) the file.

  

3. The crucial dependencies used in the project are:

-- unityagents: instructions to install: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

-- NumPy: Numerical Python library, pip installation command: 'pip install numpy'

-- Matplotlib: Python's basic plotting library, pip installation command: 'pip install matplotlib'

  -- Torch: Machine and Deep Learning library for Python, pip installation command: 'pip install torch'

The code takes form of a Jupyter Notebook. To train the agent, just run each cell one by one starting from the top.

The weights of actor and critic are saved in appropriate files.