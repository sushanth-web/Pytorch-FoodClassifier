from data_setup import create_dataloaders
from engine import train 
from model_builder import TinyVGG
from torchvision import transforms
import torch
from torch import nn
import matplotlib.pyplot as plt
from utils import save_model
import os
from custom_image import Predict_custom_image

transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = create_dataloaders( r"C:\Users\susha\OneDrive\Apps\ScriptMode\data\pizza_steak_sushi\train",
    r"C:\Users\susha\OneDrive\Apps\ScriptMode\data\pizza_steak_sushi\test",
                                                                    transforms,
                                                                    batch_size=32,
                                                                    num_workers=0)

TinyVGG_model = TinyVGG(3,10,len(class_names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TinyVGG_model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=TinyVGG_model.parameters(),lr=0.1)

results = train(TinyVGG_model, train_dataloader, test_dataloader,optimizer,loss_fn,5, device)

save_model(model=TinyVGG_model,
           target_dir="models",
           model_name="05_going_modular_script_mode_tinyvgg_model.pth")

plt.figure(figsize=(10,7))
plt.plot(results["train_loss"],label="train_loss")
plt.plot(results["test_loss"],label="test_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()

plt.figure(figsize=(10,7))
plt.plot(results["train_acc"],label="train_acc")
plt.plot(results["test_acc"],label="test_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()

plt.show()

custom_image_url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
Predict_custom_image(custom_image_url,TinyVGG_model,class_names,device)