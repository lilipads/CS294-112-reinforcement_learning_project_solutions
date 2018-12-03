import pickle
import seaborn as sns
import matplotlib.pyplot as plt
data_dir = 'q1'
timesteps1 = pickle.load(open(data_dir + '/timesteps.pkl', 'rb'))
mean_reward1 = pickle.load(open(data_dir + '/mean_reward.pkl', 'rb'))
best_mean_reward1 = pickle.load(open(data_dir + '/best_mean_reward.pkl', 'rb'))

data_dir = 'q2_double_q'
timesteps = pickle.load(open(data_dir + '/timesteps.pkl', 'rb'))
mean_reward = pickle.load(open(data_dir + '/mean_reward.pkl', 'rb'))
best_mean_reward = pickle.load(open(data_dir + '/best_mean_reward.pkl', 'rb'))

plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))
p = plt.plot(timesteps, mean_reward, label="duoble learning")
p = plt.plot(timesteps1, mean_reward1, label="without double q")
plt.xlabel('Timestep')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()

plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))
sns.set(style="darkgrid", font_scale=1.5)
p = sns.tsplot(data=best_mean_reward, time=timesteps)
plt.xlabel('Timestep')
plt.ylabel('Best Mean Reward')
plt.show()

