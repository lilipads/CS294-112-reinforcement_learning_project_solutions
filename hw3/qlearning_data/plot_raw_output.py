import pickle
import seaborn as sns
import matplotlib.pyplot as plt
data_dir = 'q2_double_q'
timesteps = pickle.load(open(data_dir + '/timesteps.pkl', 'rb'))
mean_reward = pickle.load(open(data_dir + '/mean_reward.pkl', 'rb'))
best_mean_reward = pickle.load(open(data_dir + '/best_mean_reward.pkl', 'rb'))

plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))
sns.set(style="darkgrid", font_scale=1.5)
p = sns.tsplot(data=mean_reward, time=timesteps)
plt.xlabel('Timestep')
plt.ylabel('Mean Reward')
plt.show()

plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))
sns.set(style="darkgrid", font_scale=1.5)
p = sns.tsplot(data=best_mean_reward, time=timesteps)
plt.xlabel('Timestep')
plt.ylabel('Best Mean Reward')
plt.show()

