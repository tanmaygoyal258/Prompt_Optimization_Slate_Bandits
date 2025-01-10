import numpy as np
import os
import matplotlib.pyplot as plt

TRAIN_SET_LENGTH = 3636
VAL_SET_LENGTH = 408   
TEST_SET_LENGTH = 1725

folder = "glue_mrpc_result/09-01_14-55"
for x in os.listdir(folder):
   if 'parameters' in x and not os.path.isfile(x):
    current_parameter_folder = os.path.join(folder , x)
try:
   reward = np.load(current_parameter_folder + "/rewards_array.npy")
except:
    reward = np.load(folder + "/rewards_array_final.npy")

total_length = reward.shape[0]

# overall accuracy
print("Accuracy: {:.2f}".format( reward.sum()/total_length * 100))

# accuracy for train, validation and test dataset
if total_length > TRAIN_SET_LENGTH + VAL_SET_LENGTH:
    if VAL_SET_LENGTH > 0:
        print("Accuracy for validation dataset: {:.2f}".format( reward[TRAIN_SET_LENGTH:TRAIN_SET_LENGTH+VAL_SET_LENGTH].sum()/VAL_SET_LENGTH * 100))
    print("Accuracy for test dataset: {:.2f}".format( reward[TRAIN_SET_LENGTH+VAL_SET_LENGTH:].sum()/len(reward[TRAIN_SET_LENGTH+VAL_SET_LENGTH:]) * 100))
    if TRAIN_SET_LENGTH > 0:
        print("Accuracy for train dataset: {:.2f}".format( reward[:TRAIN_SET_LENGTH].sum()/TRAIN_SET_LENGTH * 100))
    print("Accuracy for validation and test dataset: {:.2f}".format( reward[TRAIN_SET_LENGTH:].sum()/len(reward[TRAIN_SET_LENGTH:]) * 100))
elif total_length > TRAIN_SET_LENGTH:
    if VAL_SET_LENGTH > 0:
        print("Accuracy for validation dataset: {:.2f}".format( reward[TRAIN_SET_LENGTH:].sum()/len(reward[TRAIN_SET_LENGTH:]) * 100))
    if TRAIN_SET_LENGTH > 0:
        print("Accuracy for train dataset: {:.2f}".format( reward[:TRAIN_SET_LENGTH].sum()/TRAIN_SET_LENGTH * 100))

# plotting the curves
plt.figure(figsize = (20,15))

plt.subplot(1,2,1)
if reward.shape[0] <= TRAIN_SET_LENGTH:
    cum_reward = np.cumsum(reward)
    plt.plot([i for i in range(len(cum_reward))] , cum_reward , 'r-')
    plt.title("Cumulative Reward Curve")
    plt.grid(True)
else:
    val_test_reward = reward[TRAIN_SET_LENGTH:]
    accuracy = [val_test_reward.cumsum()[i]/(i+1)*100 for i in range(len(val_test_reward))]
    plt.plot([i for i in range(len(accuracy))] , accuracy , 'r-')
    plt.title("Accuracy Curve for Validation and Test Dataset")
    plt.grid(True)
    if len(val_test_reward) > VAL_SET_LENGTH:
        plt.axvline(x = VAL_SET_LENGTH , color = 'b')

plt.subplot(1,2,2)
cum_reward = np.cumsum(reward)
accuracy = [cum_reward[i]/(i+1)*100 for i in range(len(cum_reward))]
plt.plot([i for i in range(len(accuracy))] , accuracy , 'r-')
plt.title("Accuracy Curve")
plt.grid(True)
if reward.shape[0] > TRAIN_SET_LENGTH + VAL_SET_LENGTH:
    plt.axvline(x = TRAIN_SET_LENGTH , color = 'b')
    plt.axvline(x = TRAIN_SET_LENGTH + VAL_SET_LENGTH , color = 'b')
elif reward.shape[0] > VAL_SET_LENGTH:
    plt.axvline(x = TRAIN_SET_LENGTH , color = 'b')
plt.savefig(folder + "/accuracy_reward_curves.png")