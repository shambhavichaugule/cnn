import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import io

# Class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Must redefine CNN class — same as in cnn.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

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

# Load test data
print("Loading test data...")
dataset = load_dataset("uoft-cs/cifar10")

def extract_data(split):
    images = []
    labels = []
    for item in split:
        img = np.array(item['img'])
        images.append(img)
        labels.append(item['label'])
    return np.array(images), np.array(labels)

X_test, y_test = extract_data(dataset['test'])
X_test = X_test / 255.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_test_tensor = torch.FloatTensor(X_test.transpose(0, 3, 1, 2)).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Load saved model
print("Loading model...")
model = CNN().to(device)
model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
model.eval()
print("Model loaded successfully!")

# Evaluate in batches
print("Evaluating...")
all_preds = []
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

test_pred = np.array(all_preds)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=classes))