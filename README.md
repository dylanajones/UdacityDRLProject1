# UdacityDRLProject1
This is the repository for the first Udacity Deep Reinforcement Learning Project - Navigation

# Environment
In this project the agent must be trained to collect yellow bananas while avoiding blue bananas. Each time a yellow banana is collected the agent recieves +1 reward while collecting a blue banana provides the agent with -1 reward.

The state space is a 37 dimensional vector which contains information about the agents velocity as well as ray-based perception of objects around the agents forward direction.

The agent has 4 actions available to it:
- Move forward
- Move backward
- Turn left
- Turn right

This task is episodic and considered solved when an average score of +13 is achieved over 100 consecutive episodes.

# Getting Started

1. Installing needed software

Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. These instructions will have you install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

2. Getting the Environment

You will need to download the environment from one of the links below which matches your system. The default behavior in Report.ipynb is to be run in the Udacity virtual workspace where this is not required.

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

# Instructions
Follow the instructions in Report.ipynb to create and train an agent.
