import warnings
warnings.filterwarnings("ignore")

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
HF_Token = user_secrets.get_secret("HF_Token")

# from dotenv import load_dotenv
# import os
# from huggingface_hub import login
# load_dotenv()
# login(token=os.getenv("HF_Token"))


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

# ── DATA AUGMENTATION ─────────────────────────────────────
print("\n" + "="*50)
print("DATA AUGMENTATION")
print("="*50)

from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Custom dataset class with augmentation
class CIFAR10Augmented(Dataset):
    def __init__(self, images, labels, augment=True):
        self.images = images
        self.labels = labels
        self.augment = augment

        # Augmentation pipeline — applied during training
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        # No augmentation for test — just normalize
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to uint8 for PIL
        image = (image * 255).astype(np.uint8)

        if self.augment:
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)

        return image, label

# Create augmented datasets
train_aug = CIFAR10Augmented(X_train, y_train, augment=True)
test_aug = CIFAR10Augmented(X_test, y_test, augment=False)

# Create dataloaders
train_loader_aug = DataLoader(train_aug, batch_size=128, shuffle=True, num_workers=2)
test_loader_aug = DataLoader(test_aug, batch_size=128, shuffle=False, num_workers=2)

print("Augmented dataset ready!")
print(f"Training samples: {len(train_aug)}")
print(f"Test samples: {len(test_aug)}")

# Visualize augmentation effect
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# Original images
for i in range(8):
    axes[0, i].imshow(X_train[i])
    axes[0, i].set_title(classes[y_train[i]], fontsize=7)
    axes[0, i].axis('off')

# Augmented versions
for i in range(8):
    img, label = train_aug[i]
    # Denormalize for display
    img = img.numpy().transpose(1, 2, 0)
    img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
    img = np.clip(img, 0, 1)
    axes[1, i].imshow(img)
    axes[1, i].set_title(classes[label], fontsize=7)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=9)
axes[1, 0].set_ylabel('Augmented', fontsize=9)
plt.suptitle('Data Augmentation — Original vs Augmented')
plt.tight_layout()
plt.savefig('augmentation_samples.png')
plt.show()
print("Augmentation samples saved!")

# Same CNN architecture as before
class CNN_Aug(nn.Module):
    def __init__(self):
        super(CNN_Aug, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# Train with augmentation
model_aug = CNN_Aug().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_aug = optim.Adam(model_aug.parameters(), lr=0.001)
scheduler_aug = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_aug, patience=2, factor=0.5
)

print("Training with Data Augmentation...")
epochs = 40
aug_losses = []
aug_accs = []

for epoch in range(epochs):
    model_aug.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader_aug:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer_aug.zero_grad()
        outputs = model_aug(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_aug.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / len(train_loader_aug)
    avg_acc = correct / total
    aug_losses.append(avg_loss)
    aug_accs.append(avg_acc)
    scheduler_aug.step(avg_loss)

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

# Evaluate
model_aug.eval()
all_preds = []
with torch.no_grad():
    for batch_x, batch_y in test_loader_aug:
        batch_x = batch_x.to(device)
        outputs = model_aug(batch_x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

aug_pred = np.array(all_preds)
acc_aug = accuracy_score(y_test, aug_pred)

print(f"\nWithout Augmentation: 82.14%")
print(f"With Augmentation:    {acc_aug:.4f}")
print(f"Improvement:          {acc_aug - 0.8214:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, aug_pred, target_names=classes))

# Save model
torch.save(model_aug.state_dict(), 'cnn_augmented.pth')
print("\nAugmented model saved!")

# ── TRANSFER LEARNING ─────────────────────────────────────
print("\n" + "="*50)
print("TRANSFER LEARNING — ResNet18")
print("="*50)

import torchvision.models as models

# Load pretrained ResNet18
resnet = models.resnet18(pretrained=True)

print("ResNet18 architecture:")
print(f"Total parameters: {sum(p.numel() for p in resnet.parameters()):,}")

# Freeze all layers — don't update pretrained weights
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer with CIFAR-10 classifier
# ResNet18 final layer outputs 512 features
resnet.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

# Only the new final layer has requires_grad=True
trainable_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in resnet.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters:    {total_params - trainable_params:,}")
print(f"Training only {trainable_params/total_params:.1%} of the network")

resnet = resnet.to(device)

# Train ResNet with augmented data
criterion = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(
    resnet.fc.parameters(),  # only train the new final layer
    lr=0.001
)
scheduler_resnet = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_resnet, patience=3, factor=0.5
)

print("Training ResNet18 (transfer learning)...")
epochs = 20
resnet_losses = []
resnet_accs = []

for epoch in range(epochs):
    resnet.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader_aug:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer_resnet.zero_grad()
        outputs = resnet(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_resnet.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / len(train_loader_aug)
    avg_acc = correct / total
    resnet_losses.append(avg_loss)
    resnet_accs.append(avg_acc)
    scheduler_resnet.step(avg_loss)

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

# Evaluate
resnet.eval()
all_preds = []
with torch.no_grad():
    for batch_x, batch_y in test_loader_aug:
        batch_x = batch_x.to(device)
        outputs = resnet(batch_x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

resnet_pred = np.array(all_preds)
acc_resnet = accuracy_score(y_test, resnet_pred)

print(f"\nFinal Comparison:")
print(f"{'Model':<35} {'Accuracy':>10}")
print("-" * 47)
print(f"{'CNN from scratch':<35} {'0.8214':>10}")
print(f"{'CNN + Augmentation':<35} {acc_aug:>10.4f}")
print(f"{'ResNet18 Transfer Learning':<35} {acc_resnet:>10.4f}")

print("\nClassification Report:")
print(classification_report(y_test, resnet_pred, target_names=classes))

# Save model
torch.save(resnet.state_dict(), 'resnet_transfer.pth')
print("\nResNet model saved!")

# ── RESNET FIXED ─────────────────────────────────────────
print("\n" + "="*50)
print("RESNET18 — FIXED VERSION")
print("="*50)

# Fix 1 — Resize images to 224×224 and use ImageNet normalization
train_transform_fixed = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),              # resize to 224×224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

test_transform_fixed = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class CIFAR10Fixed(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = (self.images[idx] * 255).astype(np.uint8)
        image = self.transform(image)
        return image, self.labels[idx]

train_fixed = CIFAR10Fixed(X_train, y_train, train_transform_fixed)
test_fixed = CIFAR10Fixed(X_test, y_test, test_transform_fixed)

train_loader_fixed = DataLoader(train_fixed, batch_size=64, shuffle=True, num_workers=2)
test_loader_fixed = DataLoader(test_fixed, batch_size=64, shuffle=False, num_workers=2)

# Fix 2 — Load fresh ResNet and unfreeze last 2 layers
resnet_fixed = models.resnet18(pretrained=True)

# Unfreeze last residual block + final layer
for name, param in resnet_fixed.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace final layer
resnet_fixed.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

resnet_fixed = resnet_fixed.to(device)

trainable = sum(p.numel() for p in resnet_fixed.parameters() if p.requires_grad)
total = sum(p.numel() for p in resnet_fixed.parameters())
print(f"Trainable parameters: {trainable:,} ({trainable/total:.1%} of network)")

# Train
optimizer_fixed = optim.Adam(
    filter(lambda p: p.requires_grad, resnet_fixed.parameters()),
    lr=0.0001    # lower learning rate for fine-tuning
)
scheduler_fixed = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_fixed, patience=2, factor=0.5
)

print("\nTraining fixed ResNet18...")
epochs = 10  # fewer epochs needed — pretrained features are strong

for epoch in range(epochs):
    resnet_fixed.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader_fixed:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer_fixed.zero_grad()
        outputs = resnet_fixed(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_fixed.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / len(train_loader_fixed)
    avg_acc = correct / total
    scheduler_fixed.step(avg_loss)

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

# Evaluate
resnet_fixed.eval()
all_preds = []
with torch.no_grad():
    for batch_x, batch_y in test_loader_fixed:
        batch_x = batch_x.to(device)
        outputs = resnet_fixed(batch_x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

resnet_fixed_pred = np.array(all_preds)
acc_resnet_fixed = accuracy_score(y_test, resnet_fixed_pred)

print(f"\nFinal Comparison:")
print(f"{'Model':<35} {'Accuracy':>10}")
print("-" * 47)
print(f"{'CNN from scratch':<35} {'0.8214':>10}")
print(f"{'CNN + Augmentation (20 epochs)':<35} {acc_aug:>10.4f}")
print(f"{'ResNet18 (wrong config)':<35} {acc_resnet:>10.4f}")
print(f"{'ResNet18 (fixed)':<35} {acc_resnet_fixed:>10.4f}")

print("\nClassification Report:")
print(classification_report(y_test, resnet_fixed_pred, target_names=classes))

torch.save(resnet_fixed.state_dict(), 'resnet_fixed.pth')
print("\nFixed ResNet model saved!")
