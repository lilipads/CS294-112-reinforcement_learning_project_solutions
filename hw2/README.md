# CS294-112 HW 2: Policy Gradient

## Background
In this project, I implemented policy gradient algorithm for reinforcement learning and qualitatively compared a few variants of it: adding reward to go (reward to go), normalize the reward (normalize), subtracting the reward by different baselines (baseline) and a baseline that is predicted with a neural network (baseline_nn). The purpose of all these variants are to reduce the variance and improve the model.

## Method
Read `train_pg_f18.py` for implmentation of the model. 

![computational_graph](img/graph.png)

The figure above illustrates the computational graph of the model. We run two types of loops on this graph, illustrated by the orange arrows and gray arrows. 

First is the orange loop, which involves both tensorflow and the simulator (dashed lines).  We feed an observation into our network which generates a distrbution of actions; we sample an action from the distribution. We execute that action in the simualator which in turn gives us a new observation and a reward. We repeat the cycle multiple times and collect the set of observations, actions and rewards.

Next, we move onto the gray loop. We feed the network with the data we just collected. This time, we know the reward, so we can use it to calculate the advantage and eventually the loss function. This way, we can backpropagate and update the variables in the neural network.

There is one more piece not shown in the above graph. Later on, we add one more component: instead of using the average as the baseline, we build a neural network to predict the reward given observation. The training data is the reward and observation we collected from the simulator so far.

Since the computational graph for this algorithm is more complex than the usual deep learning pipelines for supervised learning, here are some useful debugging tips:

* Be really careful about the tensor dimensions. Sometimes we will encounter silent failures when the dimension does not match but tensorflow does multiplication by broadcasting.
* For the tf.multinomial and tf.sparse_softmax_cross_entropy_with_logits, the arguments are _unscaled_ logit (which can range from -inf to +inf). So, if you are using those two functions, don't apply softmax on the last layer of the neural network!
* Watch out for the sign of the loss. log Pi * reward is to be _maximized_. Later, in the Adam optimizer, we asked it to minimize. In the discrete case, when we apply cross entropy on it, it is calculating - log Pi. So the two negation cancels out. In the continuous case, however, we do need to add a minus sign before the log probability for the term to be minimized.
* Unlike supervised learning, loss does not necessarily go down over time. In fact it likely goes up because as we perform better, the rewards go up and they weigh the samples more. See below for a plot of loss generated over the course of training. So we can't use the old trick to debug by overfitting on a small sample and checking loss going to zero. Instead, we should be monitoring the reward and expect to see it go up.
![loss](img/loss.png)

## Results

We first tested on the Carpole task. The goal is to move the cart in such a way that the pole does not fall down. See the gif for the learned behavior on our trained model:

![learned](img/learned.gif)

Adding reward-to-go and normalization progressively improves the performance of the model as reflected by the larger average reward. Notice also how the variance shrinks too.

![compare_features](img/compare_features.png)

Making the batch size larger (collect more samples from the simulator in between each training epoch) also makes the model to converge faster. Here, a small batch is 1000 runs and a large batch is 5000 runs.

![compare_batch](img/compare_batch.png)

However, the advantage diminishes in this task when we use reward-to-go along as normalization, as the small batch size (1000) already converges quite fast.

![diminish_advantage](img/diminish_advantage.png)


## Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

See the [HW2 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) for further instructions.
