import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time


# Define the CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out


# Load the MNIST dataset
train_data = MNIST(root='data', train=True, transform=ToTensor(), download=True)
test_data = MNIST(root='data', train=False, transform=ToTensor(), download=True)

# Set up the data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Set up the model and optimizer
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 50
train_loss, test_loss = [], []
train_acc, test_acc = [], []

start = time()
for epoch in range(num_epochs):
    # Train the model for one epoch
    running_train_loss, running_train_acc = 0.0, 0.0
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
        running_train_acc += (outputs.argmax(dim=1) == labels).sum().item()
    epoch_train_loss = running_train_loss / len(train_data)
    epoch_train_acc = running_train_acc / len(train_data)

    # Evaluate the model on the test set
    running_test_loss, running_test_acc = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * images.size(0)
            running_test_acc += (outputs.argmax(dim=1) == labels).sum().item()
    epoch_test_loss = running_test_loss / len(test_data)
    epoch_test_acc = running_test_acc / len(test_data)

    # Record the metrics for this epoch
    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    if (epoch + 1) % 10 == 0:
        print("Epoch %d, Train Loss: %.3f, Train Acc: %.2f%%, Test Loss: %.3f, Test Acc: %.2f%%" %
              (epoch + 1, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc))

stop = time()

# Plot the train and test losses
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax[0].plot(train_loss, label='Train')
ax[0].plot(test_loss, label='Test')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Train/Test Losses')
ax[0].legend()

# Plot the train and test accuracies
ax[1].plot(train_acc, label='Train')
ax[1].plot(test_acc, label='Test')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title('Train/Test Accuracies')
ax[1].legend()

print(f"time passed: {stop - start} seconds")
plt.show()
