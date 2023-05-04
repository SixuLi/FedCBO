import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def get_selection_SR_from_txt(file_name):
    with open(file_name, 'r') as file:
        contents = file.readlines()

    # Creat an array to store the selection successful rate for each communication round.
    selection_SR = np.zeros(100)

    for line in contents:
        parts = line.split(":")
        if parts[0] == "Communication round":
            # Check communication round idx.
            communication_idx = int(parts[1].strip())

            if communication_idx == 0:
                num_correct_selection = 0
                num_total_selection = 0
            else:
                # Store the successful rate for the previous round.
                selection_SR[communication_idx-1] = num_correct_selection / num_total_selection
                num_correct_selection = 0
                num_total_selection = 0
        else:
            # Get the number of correct selections and total number of selections for each agent.
            numerator, denominator = parts[1].strip().split("/")
            num_correct_selection += int(numerator)
            num_total_selection += int(denominator)

    # Store the successful rate for the last round.
    selection_SR[communication_idx] = num_correct_selection / num_total_selection

    return selection_SR

selection_SR = np.zeros((5,100))
for seed_idx in range(5):
    file_name = '../results/FedCBO_Rotated_MNIST/Seed_{}/check_state.txt'.format(seed_idx)
    selection_SR[seed_idx] = get_selection_SR_from_txt(file_name)
#print(selection_SR)

def get_selection_rate(epsilon, epsilon_decay, T):
    selection_rate = np.zeros(T)
    for t in range(T):
        selection_rate[t] = 1 - max((epsilon - epsilon_decay*t), 0.09)*0.75
    return selection_rate

deterministic_SR = get_selection_rate(epsilon=0.5, epsilon_decay=0.01, T=100)

iterations = np.arange(100)

# Compute the mean and std of the selection successful rate over 5 random seeds.
SR_mean = np.mean(selection_SR, axis=0)
SR_std = np.std(selection_SR, axis=0)
print(SR_mean)
print(SR_std)

# Compute the upper and lower bounds of the standard deviation
SR_upper = SR_mean + SR_std
SR_lower = SR_mean - SR_std

plt.figure(figsize=(16,9), dpi=120)

# Plot the curve with the standard deviation
plt.plot(iterations, SR_mean, label='Empirical average successful SR')
# plt.fill_between(iterations, SR_lower, SR_upper,
#                  color='lightblue', alpha=0.5, label='Standard Deviation')
plt.plot(iterations, deterministic_SR, label='Oracle expected successful SR')

# Add legend and labels
plt.legend(loc=4, prop={'size': 20})
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Average Selection Rate (SR)', fontsize=20)
plt.savefig('../results/FedCBO_Rotated_MNIST/SR.png')
plt.show()



#print(selection_SR)

#plt.plot(iterations, selection_SR)
#plt.plot(iterations, deterministic_SR)

#plt.show()





