# Download custom image
import requests
import torchvision
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def Predict_custom_image(url:str,model_1:torch.nn.Module,class_names:list[str],device:torch.device):
     
    custom_image_path = Path(r"C:\Users\susha\OneDrive\Apps\ScriptMode\data") / url.split("/")[-1] # get the last part of the URL as the filename

    # Download the image if it doesn't already exist
    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get(url)
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")

    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    custom_image = custom_image / 255

    # Create transform pipleine to resize image
    custom_image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    # Transform target image
    custom_image_transformed = custom_image_transform(custom_image)


    model_1.eval()
    with torch.inference_mode():

        custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))

    # Print out prediction logits
    print(f"Prediction logits: {custom_image_pred}")

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    print(f"Prediction probabilities: {custom_image_pred_probs}")

    # Convert prediction probabilities -> prediction labels
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    print(f"Prediction label: {custom_image_pred_label}")

    print(f"Predicted class name: {class_names[custom_image_pred_label]}")

    plt.imshow(custom_image_transformed.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[custom_image_pred_label.cpu()]} | Prob: {custom_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {custom_image_pred_label} | Prob: {custom_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False);