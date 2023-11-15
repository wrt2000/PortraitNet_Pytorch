import matplotlib.pyplot as plt
import re
# log_file_path = "results/eg1800_20231114161557/training_log.log"
log_file_path = "results/Supervisely_face_20231113221803/training_log.log"

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
                epoch = int(numbers[7])
                train_loss = float(numbers[8])
                val_loss = float(numbers[9])
                iou = float(numbers[10])

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
plt.title('Training and Validation Loss per Epoch on Supervisely_face Dataset')  # Supervisely_face
plt.legend()
plt.grid(True)
plt.savefig('Superviseloss.png')

# Plotting IoU vs epochs
plt.figure(figsize=(12, 6))
plt.plot(epoch_numbers, ious, label='IoU', color='green')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('IoU per Epoch on Supervisely_face Dataset')
plt.legend()
plt.grid(True)
plt.savefig('Superviseiou.png')
