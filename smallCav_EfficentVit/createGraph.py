import matplotlib.pyplot as plt
import re

log_data = """
Epoch 1 train loss 1.237 val loss 0.934 val_mIoU 34.09%
Epoch 2 train loss 0.676 val loss 0.484 val_mIoU 63.04%
...
Epoch 419 train loss 0.010 val loss 0.114 val_mIoU 95.38%
Epoch 420 train loss 0.009 val loss 0.117 val_mIoU 95.37%
"""

# Parse the log data
epochs = []
val_mIoU = []

for line in log_data.strip().split('\n'):
    epoch_match = re.search(r'Epoch (\d+)', line)
    val_mIoU_match = re.search(r'val_mIoU (\d+\.\d+)%', line)
    
    if epoch_match and val_mIoU_match:
        epochs.append(int(epoch_match.group(1)))
        val_mIoU.append(float(val_mIoU_match.group(1)))

# Plot the data
plt.figure(figsize=(12, 8))
plt.plot(epochs, val_mIoU, label='Validation mIoU')

plt.xlabel('Epochs')
plt.ylabel('mIoU (%)')
plt.title('Small CAV on EfficentVit (3500 Images)')
plt.legend()
plt.grid(True)

# Save the plot as an SVG file
plt.savefig('cav_efficentvit_ethan.svg', format='svg')
plt.show()
