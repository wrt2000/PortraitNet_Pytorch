import matplotlib.pyplot as plt
import re
log_file_path = "/workspace/wangruotong/PortraitNet/results/eg1800_20231114161557/training_log.log"

epoch_numbers = []
train_losses = []
val_losses = []
ious = []

with open(log_file_path, 'r') as file:
    for line in file:
        if 'Train Loss' in line:
            # Use regular expressions to find the numerical values
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(numbers) >= 4:
                # Convert extracted strings to appropriate types
                epoch = int(numbers[0])
                train_loss = float(numbers[1])
                val_loss = float(numbers[2])
                iou = float(numbers[3])

                # Append the data to the lists
                epoch_numbers.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                ious.append(iou)

# Plotting training loss and validation loss vs epochs
plt.figure(figsize=(12, 6))
plt.plot(epoch_numbers, train_losses, label='Training Loss')
plt.plot(epoch_numbers, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch on EG1800 Dataset')
plt.legend()
plt.grid(True)
plt.savefig('EG1800loss.png')

# Plotting IoU vs epochs
plt.figure(figsize=(12, 6))
plt.plot(epoch_numbers, ious, label='IoU', color='green')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('IoU per Epoch on EG1800 Dataset')
plt.legend()
plt.grid(True)
plt.savefig('EG1800iou.png')
