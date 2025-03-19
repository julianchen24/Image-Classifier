# python predict.py digit.png

import torch
import torchvision.transforms as transforms
from PIL import Image

from classification_model import Model

model = Model(num_classes=10)
state_dict = torch.load("trained_model.pth")

# Remove "module." prefix from keys if necessary
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

image_path = "digit.png"
image = Image.open(image_path).convert("L") 

transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
])
image = transform(image).unsqueeze(0)  
# Make prediction
with torch.no_grad():
    output = model(image)
    predicted_digit = torch.argmax(output, dim=1).item()

print(f"Predicted Digit: {predicted_digit}")
