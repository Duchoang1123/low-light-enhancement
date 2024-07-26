import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

# Path to your event file
event_file_path = 'events.out.tfevents.1721821271.n-62-31-8.22552.0.v2'

# Initialize the event accumulator
event_acc = EventAccumulator(event_file_path)

# Load the event file
event_acc.Reload()

# Plot tensor values
if 'tensors' in event_acc.Tags():
    tensor_tags = event_acc.Tags()['tensors']
    for tag in tensor_tags:
        steps = []
        values = []
        tensor_events = event_acc.Tensors(tag)
        for event in tensor_events:
            # Decode the tensor proto to get the value
            tensor_value = tf.make_ndarray(event.tensor_proto)
            # steps.append(event.step)
            values.append(tensor_value.item())  # Assuming tensor_value is a scalar

        # Plotting the values
        steps = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
        if tag == 'gen_total_loss':
            gen_total_loss = values
        if tag == 'gen_gan_loss':
            gen_gan_loss = values
        if tag == 'gen_l1_loss':
            gen_l1_loss = values
        if tag == 'disc_loss':
            disc_loss = values
        if tag == 'RMSE':
            RMSE = values
        if tag == 'MRAE':
            MRAE = values
        if tag == 'PSNR':
            PSNR = values
        plt.figure()
        plt.plot(steps, values, label=tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title(f'{tag}')
        plt.legend()
        plt.grid()
        plt.savefig(f'{tag}.png')



        #plt.show()

# Create the figure and the first axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the first dataset
ax1.plot(steps, disc_loss, label='disc_loss', color='tab:blue')
ax1.plot(steps, gen_l1_loss, label='gen_l1_loss', color='tab:orange')
ax1.plot(steps, RMSE, label='RMSE', color='tab:green')

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(steps, gen_gan_loss, label='gen_gan_loss', color='tab:red')
ax2.plot(steps, gen_total_loss, label='gen_total_loss', color='tab:purple')
ax2.plot(steps, MRAE, label='MRAE', color='tab:brown')
ax2.plot(steps, PSNR, label='PSNR', color='tab:pink')

# Set labels
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss / RMSE')
ax2.set_ylabel('GAN Loss / MRAE / PSNR')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Show the plot
plt.title('Combined Metrics Over Training Steps')
plt.savefig('combined_metrics.png')
plt.show()