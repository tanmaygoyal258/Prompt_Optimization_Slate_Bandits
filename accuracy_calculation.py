import numpy as np
import os
import matplotlib.pyplot as plt

TEST_SET_LENGTH = 1000

folder = "glue_sst2_result/26-12_00-16"
for x in os.listdir(folder):
   if 'parameters' in x and not os.path.isfile(x):
    current_parameter_folder = os.path.join(folder , x)
try:
   reward = np.load(current_parameter_folder + "/rewards_array.npy")
except:
    reward = np.load(folder + "/rewards_array_final.npy")


print("Accuracy: {:.2f}".format( reward.sum()/reward.shape[0] * 100))
if reward.shape[0] > TEST_SET_LENGTH:
   print("Accuracy for test dataset: {:.2f}".format( reward[-TEST_SET_LENGTH:].sum()/TEST_SET_LENGTH * 100))


plt.figure(figsize = (20,15))
plt.subplot(1,2,1)
if reward.shape[0] < TEST_SET_LENGTH:
    cum_reward = np.cumsum(reward)
    plt.plot([i for i in range(len(cum_reward))] , cum_reward , 'r-')
    plt.title("Cumulative Reward Curve")
    plt.grid(True)
else:
    test_reward = reward[-TEST_SET_LENGTH:]
    accuracy = [test_reward.cumsum()[i]/(i+1)*100 for i in range(len(test_reward))]
    plt.plot([i for i in range(len(accuracy))] , accuracy , 'r-')
    plt.title("Accuracy Curve for Test Dataset")
    plt.grid(True)


plt.subplot(1,2,2)
cum_reward = np.cumsum(reward)
accuracy = [cum_reward[i]/(i+1)*100 for i in range(len(cum_reward))]
plt.plot([i for i in range(len(accuracy))] , accuracy , 'r-')
plt.title("Accuracy Curve")
plt.grid(True)
if reward.shape[0] >= TEST_SET_LENGTH:
    plt.axvline(x = reward.shape[0] - TEST_SET_LENGTH , color = 'b')
plt.savefig(folder + "/accuracy_reward_curves.png")



