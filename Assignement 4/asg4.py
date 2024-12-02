import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation, RandomAffine


wandb.init(project="mnist-hyperparameter-sweep", config={
    "hidden1": 128,
    "hidden2": 64,
    "dropout": 0.2,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 100,
    "epochs": 50
})

config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = Compose([
    RandomRotation(10),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

test_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

train_dataset = MNIST('data', train=True, transform=train_transforms, download=True)
test_dataset = MNIST('data', train=False, transform=test_transforms, download=True)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

class FullyConnectedNet(nn.Module):
    def __init__(self, hidden1, hidden2, dropout_rate):
        super(FullyConnectedNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

model = FullyConnectedNet(config.hidden1, config.hidden2, config.dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


training_results = []

epochs = config.epochs
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    wandb.log({"epoch": epoch + 1, "loss": loss.item()})
    training_results.append({"epoch": epoch + 1, "loss": loss.item()})
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


model.eval()
predictions = []
with torch.no_grad():
    for idx, (images, _) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())


submission_df = pd.DataFrame({
    'ID': range(len(predictions)),
    'target': predictions
})


submission_df.to_csv('submission.csv', index=False)


correct = sum([1 for label, prediction in zip(test_dataset.targets, predictions) if label == prediction])
accuracy = (correct / len(test_dataset.targets)) * 100 if len(test_dataset.targets) > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
wandb.log({"accuracy": accuracy})


wandb.finish()
