# Reinforcement Learning Projects
This is my solution (WIP) to Berkley's Deep Reinforcement Learning Class CS294-112 in Fall 2018. It consists of five projects, each of which is an implementation of a reinforcement learning algorithm. I am continuously updating this repository as I go along the course. Below are the highlights of the results:

## hw1: Behavorial Cloning
In this project, I implemented the vanilla behaviorial cloning along with the DAgger variant to succesfully train a few MuJoCo tasks. Example:

![humanoid](hw1/img/humanoid_dagger.gif)

## hw2: Policy Gradient
In this project, I implemented policy gradient algorithm and qualitatively compared four variants of it: adding reward to go (reward to go), normalize the reward (normalization), subtracting the reward by a simple baseline of the average (baseline) and a baseline that is predicted by a neural network (baseline_nn). 

Computation graph:

![graph](hw2/img/graph.png)

Learned task: Cart Pole

![cartpole](hw2/img/learned.gif)

## hw3: Q-learning and actor-critic
In this project, I implemented Q learning and compared two variants of it: with and without double Q learning.

Pong learned with Q learning (green is the trained agent):

![pong](hw3/img/pong.gif)

I also implemented actor-critic and compared its performance with policy gradient:

![actor-critic](hw3/img/inverted_pendulum_10_10.png)

## hw4: model-based RL
In this project, I implemented model-based RL. The model predicts the next state based on the current state and action. It does reasonably well as shown in the plot below:

![q1](hw4/data/HalfCheetah_q1_12-12-2018_00-10-12/prediction_009.jpg)

Results from model-based RL on the half-cheetah task (with an additional handcrafted cost function):

![model-based](hw4/img/onpolicy_rollout.png)


## hw5a: exploration
In this project, I implemented three different ways of exploration and applied it on an actor-critic algorithm to modify its reward: histogram, RBF and expamplar.

Here is a schema for one of the examplar model:

![schema](hw5/exp/img/schema.png)

We first tested our results in a PointMass environment. The agent starts off at coordinate (2, 2) in a grid world and the goal is to reach (18, 18). The reward is extremely sparse, which makes it a suitable setup for using exploration: you only get a reward for reaching the goal and 0 otherwise. Here is the visualization of the trained agent:

No exploration:

![no_exploration_gif](hw5/exp/data/ac_PM_bc0_s8_PointMass-v0_14-12-2018_20-40-50/1/exploration.gif)

Exploration using histogram:

![histogram_gif](hw5/exp/data/ac_PM_hist_bc0.01_s8_PointMass-v0_14-12-2018_16-52-20/1/exploration.gif)

As shown, when not using exploration, the agent does not branch out to try different states as often, and therefore has a small chance of hitting the goal state. When using exploration, once the agent hits the goal state and gets the reward, it learns to go straight to that state.



* Note: These are partial solutions. I didn't post results for some problems, such as hyper parameter tuning reports.
