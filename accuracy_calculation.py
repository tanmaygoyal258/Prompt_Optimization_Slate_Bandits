import numpy as np
import os
import matplotlib.pyplot as plt


folder = "glue_sst2_result/16-12_18-00"
for x in os.listdir(folder):
   if 'parameters' in x and not os.path.isfile(x):
    current_parameter_folder = os.path.join(folder , x)
try:
   reward = np.load(current_parameter_folder + "/rewards_array.npy")
except:
    reward = np.load(folder + "/rewards_array_final.npy")

print("Accuracy: {:.2f}".format( reward.sum()/reward.shape[0] * 100))

plt.figure(figsize = (20,15))
plt.subplot(1,2,1)
cum_reward = np.cumsum(reward)
plt.plot([i for i in range(len(cum_reward))] , cum_reward , 'r-')
plt.title("Cumulative Reward Curve")
plt.grid(True)

plt.subplot(1,2,2)
accuracy = [cum_reward[i]/(i+1)*100 for i in range(len(cum_reward))]
plt.plot([i for i in range(len(accuracy))] , accuracy , 'r-')
plt.title("Accuracy Curve")
plt.grid(True)
plt.savefig(folder + "/accuracy_reward_curves.png")
