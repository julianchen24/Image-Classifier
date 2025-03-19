# .\venv\Scripts\activate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from composer import Trainer
from composer.models import ComposerClassifier
from composer.algorithms import LabelSmoothing, CutMix, ChannelsLast

class Model(nn.Module):
    """Toy convolutional neural network architecture in pytorch for MNIST."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=0)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (4, 4))
        out = torch.flatten(out, 1, -1)
        out = self.fc1(out)
        out = F.relu(out)
        return self.fc2(out)

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=128)

val_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=128)

trainer = Trainer(
    model=ComposerClassifier(module=Model(), num_classes=10),
    train_dataloader=train_dataloader,
    max_duration="2ep",
    algorithms=[
        LabelSmoothing(smoothing=0.1),
        CutMix(alpha=1.0),
        ChannelsLast(),
    ],
    device="cpu"
)
trainer.fit()

# Evaluate the model on validation set
trainer.eval(eval_dataloader=val_dataloader)

# Extract and print evaluation metrics
eval_metrics = trainer.state.eval_metrics
cross_entropy = eval_metrics["eval"]["CrossEntropy"].compute()
accuracy = eval_metrics["eval"]["MulticlassAccuracy"].compute()

print(f"Validation CrossEntropy Loss: {cross_entropy:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# Save the trained model
torch.save(trainer.state.model.state_dict(), "trained_model.pth")
print("Model saved successfully as 'trained_model.pth'")
