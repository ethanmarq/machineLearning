import matplotlib.pyplot as plt
import re

log_data = """
Epoch 1 train loss 1.414 val loss 1.132 val_mIoU 36.49%
Epoch 2 train loss 0.919 val loss 0.721 val_mIoU 50.69%
...
Epoch 206 train loss 0.205 val loss 0.229 val_mIoU 71.83%
Epoch 207 train loss 0.207 val loss 0.229 val_mIoU 72.04%
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
plt.title('Big CAV on EfficentVit (2700 Images)')
plt.legend()
plt.grid(True)

# Save the plot as an SVG file
plt.savefig('cav_efficentvit_ethan.svg', format='svg')
plt.show()
