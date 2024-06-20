import pandas as pd
import matplotlib.pyplot as plt

# Load the log_gaussian_scenes.csv file
log_file_path = 'log.csv'
df = pd.read_csv(log_file_path)

# Extract relevant data
train_df = df[df['data_set'] == 'train']
valid_df = df[df['data_set'] == 'valid']

epochs_train = train_df['epoch'].values
epochs_valid = valid_df['epoch'].values

# Turn epochs into ints
epochs_train = [int(epoch) for epoch in epochs_train]
epochs_valid = [int(epoch) for epoch in epochs_valid]

train_loss = train_df['loss'].values
valid_loss = valid_df['loss'].values
valid_accuracy = valid_df['accuracy'].values

# Create a plot with two y-axes
fig, ax1 = plt.subplots(figsize=(8, 3))

color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs_train, train_loss, label='Training Loss', color=color)
ax1.plot(epochs_valid, valid_loss, label='Validation Loss', color='tab:orange')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # instantiate a second y-axis that shares the same x-axis
color = 'tab:green'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(epochs_valid, valid_accuracy, label='Validation Accuracy', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')
# set max to 100
ax2.set_ylim([min(valid_accuracy), 100])

ax1.set_xticks(epochs_valid)

# fig.tight_layout()  # to ensure the right y-label is not slightly clipped
plt.title('Training and Validation Loss and Validation Accuracy')
plt.savefig('plot.png')