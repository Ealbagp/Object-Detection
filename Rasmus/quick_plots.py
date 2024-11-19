#%%
import matplotlib.pyplot as plt
import numpy as np

# Extracted values from the log
epochs = np.arange(1, 11)

# Loss values
loss_train = [0.745, 0.660, 0.635, 0.635, 0.627, 0.629, 0.632, 0.630, 0.626, 0.632]
loss_test = [1.284, 1.256, 1.258, 1.378, 1.222, 1.261, 1.219, 1.001, 1.076, 1.085]

# Accuracy values (in percentage)
accuracy_train = [76.5, 78.8, 79.4, 79.6, 79.5, 79.2, 79.4, 79.6, 79.6, 79.3]
accuracy_test = [73.7, 76.0, 75.9, 76.1, 76.3, 75.6, 75.5, 75.4, 76.4, 76.6]

# Plotting the Loss values
plt.figure(figsize=(12, 5))
plt.plot(epochs, loss_train, label='Train Loss', marker='o')
plt.plot(epochs, loss_test, label='Test Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the Accuracy values
plt.figure(figsize=(12, 5))
plt.plot(epochs, accuracy_train, label='Train Accuracy', marker='o')
plt.plot(epochs, accuracy_test, label='Test Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()
# %%