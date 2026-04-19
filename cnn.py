import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load CIFAR-10
print("Loading CIFAR-10...")
dataset = load_dataset("uoft-cs/cifar10")

# Convert to numpy arrays
def extract_data(split):
    images = []
    labels = []
    for item in split:
        img = np.array(item['img'])  # 32×32×3
        images.append(img)
        labels.append(item['label'])
    return np.array(images), np.array(labels)

print("Processing training data...")
X_train, y_train = extract_data(dataset['train'])
print("Processing test data...")
X_test, y_test = extract_data(dataset['test'])

# Normalize to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Training samples: {X_train.shape}")
print(f"Test samples: {X_test.shape}")
print(f"Image shape: {X_train[0].shape} (32×32×3 RGB)")
print(f"Classes: {classes}")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
for i in range(10):
    axes[i].imshow(X_train[i])
    axes[i].set_title(classes[y_train[i]])
    axes[i].axis('off')
plt.suptitle('CIFAR-10 Sample Images')
plt.tight_layout()
plt.savefig("sample_images.png")
print("Sample images saved!")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Convert to PyTorch tensors
# CNN expects (batch, channels, height, width) — transpose from (H,W,C) to (C,H,W)
X_train_tensor = torch.FloatTensor(X_train.transpose(0, 3, 1, 2)).to(device)
X_test_tensor = torch.FloatTensor(X_test.transpose(0, 3, 1, 2)).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Create dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

print(f"\nData ready!")
print(f"Tensor shape: {X_train_tensor.shape} (batch, channels, height, width)")

# ── CNN ARCHITECTURE ─────────────────────────────────────
print("\n" + "="*50)
print("CONVOLUTIONAL NEURAL NETWORK")
print("="*50)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 channels in, 32 filters out
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 32×32 → 16×16
            nn.Dropout2d(0.2)
        )

        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 channels in, 64 filters out
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 16×16 → 8×8
            nn.Dropout2d(0.3)
        )

        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 channels in, 128 filters out
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 8×8 → 4×4
            nn.Dropout2d(0.4)
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 128×4×4 = 2048 → flat vector
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)             # 10 classes
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# Initialize model
model = CNN().to(device)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler — reduces LR when progress plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

print("\nTraining CNN...")
epochs = 20
train_losses = []
train_accs = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / len(train_loader)
    avg_acc = correct / total
    train_losses.append(avg_loss)
    train_accs.append(avg_acc)

    # Update scheduler
    scheduler.step(avg_loss)

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

# Save model
torch.save(model.state_dict(), 'cnn_model.pth')
print("Model saved!")

