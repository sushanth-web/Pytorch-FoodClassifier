# 🍕🥩🍣 Pizza, Steak & Sushi Image Classifier

A modular **image classification project built with PyTorch** that uses a custom **TinyVGG convolutional neural network (CNN)** to classify food images into three categories:

* 🍕 Pizza
* 🥩 Steak
* 🍣 Sushi

The project demonstrates a complete deep-learning workflow, including dataset preparation, image preprocessing, data loading, CNN model building, training, evaluation, model saving, and prediction on custom images.

---

## 📌 Project Overview

This project was built to understand how an image classification system can be developed using **PyTorch** in a modular and reusable way.

Instead of keeping the entire implementation inside one notebook or script, the project separates important components into individual Python modules:

* Dataset and DataLoader creation
* Model architecture
* Training and testing loops
* Model saving
* Custom image prediction

The model is based on the **TinyVGG architecture**, using convolutional layers, ReLU activations, max-pooling, and a final linear classification layer.

---

## ✨ Features

* ✅ Multi-class image classification
* ✅ Classifies **pizza, steak, and sushi**
* ✅ Custom TinyVGG CNN architecture
* ✅ Modular PyTorch project structure
* ✅ Image preprocessing and resizing
* ✅ Data augmentation using `TrivialAugmentWide`
* ✅ PyTorch `DataLoader` for efficient batching
* ✅ Training and testing loops
* ✅ Cross-entropy loss
* ✅ SGD optimizer
* ✅ Training and testing loss tracking
* ✅ Training and testing accuracy tracking
* ✅ Model checkpoint saving using `.pth`
* ✅ Custom image prediction from a URL
* ✅ Prediction probability using Softmax
* ✅ Visualization of custom image predictions
* ✅ CPU/GPU device selection

---

## 🧠 Model Architecture

The project uses a custom implementation of **TinyVGG**.

### Architecture

```text
Input Image
   │
   │ 3 × 64 × 64
   ▼
┌─────────────────────────┐
│     Conv2D              │
│     3 → 10 channels     │
│     Kernel = 3 × 3      │
│     Stride = 1          │
└─────────────────────────┘
   │
   ▼
  ReLU
   │
   ▼
┌─────────────────────────┐
│     Conv2D              │
│     10 → 10 channels    │
│     Kernel = 3 × 3      │
└─────────────────────────┘
   │
   ▼
  ReLU
   │
   ▼
 MaxPool2D
   │
   ▼
┌─────────────────────────┐
│     Conv2D              │
│     10 → 10 channels    │
│     Kernel = 3 × 3      │
└─────────────────────────┘
   │
   ▼
  ReLU
   │
   ▼
┌─────────────────────────┐
│     Conv2D              │
│     10 → 10 channels    │
│     Kernel = 3 × 3      │
└─────────────────────────┘
   │
   ▼
  ReLU
   │
   ▼
 MaxPool2D
   │
   ▼
 Flatten
   │
   ▼
 Linear Layer
   │
   ▼
3 Output Classes
   │
   ├── Pizza
   ├── Steak
   └── Sushi
```

### Model configuration

| Parameter          |     Value |
| ------------------ | --------: |
| Input channels     |         3 |
| Image size         |   64 × 64 |
| Hidden channels    |        10 |
| Convolution kernel |     3 × 3 |
| Convolution stride |         1 |
| Pooling            | MaxPool2D |
| Activation         |      ReLU |
| Output classes     |         3 |
| Final layer        |    Linear |

---

## 📂 Project Structure

```text
ScriptMode/
│
├── data.py
│
├── data/
│   ├── 04-pizza-dad.jpeg
│   │
│   └── pizza_steak_sushi/
│       ├── train/
│       │   ├── pizza/
│       │   ├── steak/
│       │   └── sushi/
│       │
│       └── test/
│           ├── pizza/
│           ├── steak/
│           └── sushi/
│
└── going_modular/
    │
    ├── data_setup.py
    ├── engine.py
    ├── model_builder.py
    ├── train.py
    ├── utils.py
    └── custom_image.py
```

---

# 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **Matplotlib**
* **Requests**
* **tqdm**
* **Pathlib**

---

# 📊 Dataset

The project uses a three-class food image dataset containing:

```text
Pizza
Steak
Sushi
```

The images are organized using the structure expected by PyTorch's `ImageFolder`:

```text
pizza_steak_sushi/
├── train/
│   ├── pizza/
│   ├── steak/
│   └── sushi/
│
└── test/
    ├── pizza/
    ├── steak/
    └── sushi/
```

Each directory name represents a class.

The dataset is downloaded programmatically by `data.py`.

The project uses the **Pizza, Steak & Sushi dataset** from the PyTorch Deep Learning course by **Daniel Bourke**.

---

# 🔄 Data Preprocessing

Before images are passed to the model, they are resized and transformed.

The training transformation pipeline is:

```python
transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])
```

### 1. Resize

```python
transforms.Resize((64, 64))
```

Every image is resized to:

```text
64 × 64 pixels
```

This ensures that all images have the same dimensions before being passed to the CNN.

### 2. TrivialAugmentWide

```python
transforms.TrivialAugmentWide()
```

This applies random image augmentations during training.

The purpose is to expose the model to slightly different versions of training images and help improve generalization.

### 3. ToTensor

```python
transforms.ToTensor()
```

Converts images into PyTorch tensors and scales pixel values into the range:

```text
[0, 1]
```

---

# 📦 Data Loading

The project uses `torchvision.datasets.ImageFolder` to automatically create datasets from the directory structure.

```python
train_data = datasets.ImageFolder(
    train_dir,
    transform=transform
)

test_data = datasets.ImageFolder(
    test_dir,
    transform=transform
)
```

The datasets are then converted into PyTorch `DataLoader` objects.

```python
DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)
```

### Configuration

```text
Batch size: 32
Training shuffle: Yes
Testing shuffle: No
Workers: 0
```

`num_workers=0` is used for compatibility with the current Windows development environment.

---

# 🧠 TinyVGG Implementation

The model is implemented in:

```text
going_modular/model_builder.py
```

The model contains two convolutional blocks.

### First convolution block

```python
self.conv_block_1 = nn.Sequential(
    nn.Conv2d(
        in_channels=input_shape,
        out_channels=hidden_units,
        kernel_size=3,
        stride=1,
        padding=0
    ),
    nn.ReLU(),
    nn.Conv2d(
        in_channels=hidden_units,
        out_channels=hidden_units,
        kernel_size=3,
        stride=1,
        padding=0
    ),
    nn.ReLU(),
    nn.MaxPool2d(
        kernel_size=2,
        stride=2
    )
)
```

### Second convolution block

```python
self.conv_block_2 = nn.Sequential(
    nn.Conv2d(
        hidden_units,
        hidden_units,
        kernel_size=3,
        padding=0
    ),
    nn.ReLU(),
    nn.Conv2d(
        hidden_units,
        hidden_units,
        kernel_size=3,
        padding=0
    ),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

### Classifier

After the convolutional blocks, the feature maps are flattened and passed to a linear layer:

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(
        in_features=hidden_units * 13 * 13,
        out_features=output_shape
    )
)
```

The final layer produces **three logits**, one for each class.

---

# 🏋️ Training

The training pipeline is implemented in:

```text
going_modular/engine.py
```

The model is trained using:

```python
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    params=TinyVGG_model.parameters(),
    lr=0.1
)
```

### Training configuration

| Parameter     |            Value |
| ------------- | ---------------: |
| Epochs        |                5 |
| Batch size    |               32 |
| Optimizer     |              SGD |
| Learning rate |              0.1 |
| Loss function | CrossEntropyLoss |
| Image size    |          64 × 64 |
| Hidden units  |               10 |

---

# 🔁 Training Process

For every batch, the training loop performs:

```text
Input images
      ↓
Forward pass
      ↓
Model predictions
      ↓
Calculate loss
      ↓
Zero gradients
      ↓
Backpropagation
      ↓
Optimizer step
      ↓
Calculate accuracy
```

The important training operations are:

```python
y_pred = model(X)

loss = loss_fn(y_pred, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

---

# 🧪 Model Evaluation

After each training epoch, the model is evaluated on the test dataset.

During testing:

```python
model.eval()
```

and gradient calculation is disabled:

```python
with torch.inference_mode():
```

The project records:

* Training loss
* Training accuracy
* Testing loss
* Testing accuracy

These results are returned in a dictionary:

```python
results = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}
```

This allows the training process to be analyzed after completion.

---

# 📈 Training Metrics

The project also plots training and testing metrics using Matplotlib.

### Loss

```text
Training Loss
Testing Loss
```

### Accuracy

```text
Training Accuracy
Testing Accuracy
```

These graphs can be used to understand how the model learns over the training epochs and identify potential overfitting or underfitting.

---

# 💾 Saving the Model

The trained model is saved using its `state_dict`.

The utility function is located in:

```text
going_modular/utils.py
```

The model is saved as:

```text
models/
└── 05_going_modular_script_mode_tinyvgg_model.pth
```

The model is saved using:

```python
torch.save(
    obj=model.state_dict(),
    f=model_save_path
)
```

Saving the `state_dict` stores the trained model parameters rather than the entire model object.

---

# 🖼️ Custom Image Prediction

The project supports prediction on an image obtained from a URL.

The functionality is implemented in:

```text
going_modular/custom_image.py
```

The image is:

1. Downloaded
2. Read using `torchvision`
3. Converted to `float32`
4. Normalized to `[0, 1]`
5. Resized to `64 × 64`
6. Passed through the trained CNN
7. Converted into probabilities using Softmax
8. Classified using the highest probability

---

## 🔮 Prediction Pipeline

```text
Image URL
    ↓
Download Image
    ↓
Read Image
    ↓
Convert to Float32
    ↓
Scale pixels to [0,1]
    ↓
Resize to 64×64
    ↓
Add Batch Dimension
    ↓
TinyVGG
    ↓
Prediction Logits
    ↓
Softmax
    ↓
Class Probabilities
    ↓
Argmax
    ↓
Predicted Class
```

The prediction probabilities are calculated using:

```python
torch.softmax(
    custom_image_pred,
    dim=1
)
```

The predicted class is selected using:

```python
torch.argmax(
    custom_image_pred_probs,
    dim=1
)
```

---

# ⚙️ Device Support

The project automatically determines whether CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

Therefore, the model can run on:

* CPU
* NVIDIA CUDA GPU

The model is moved to the selected device before training and inference.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd ScriptMode
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

Install the required libraries:

```bash
pip install torch torchvision matplotlib requests tqdm
```

---

# 📥 Prepare the Dataset

Run:

```bash
python data.py
```

This script downloads the Pizza, Steak & Sushi dataset and extracts it into the `data` directory.

---

# ▶️ Train the Model

Navigate into the modular project:

```bash
cd going_modular
```

Run:

```bash
python train.py
```

The training script will:

1. Load the dataset
2. Create training and testing DataLoaders
3. Create the TinyVGG model
4. Select CPU/GPU
5. Train the model
6. Evaluate the model
7. Save the trained model
8. Plot loss and accuracy
9. Run a prediction on a custom image

---

# 📋 Example Training Output

The training loop produces output similar to:

```text
Epoch: 1 | train_loss: ... | train_acc: ... | test_loss: ... | test_acc: ...
Epoch: 2 | train_loss: ... | train_acc: ... | test_loss: ... | test_acc: ...
Epoch: 3 | train_loss: ... | train_acc: ... | test_loss: ... | test_acc: ...
...
```

The exact values depend on the training run, initialization, hardware, and data augmentation.

---

# 🧩 Key Deep Learning Concepts Demonstrated

This project provides practical experience with several important deep-learning concepts:

### Convolution

CNN convolutional layers learn visual features such as:

```text
Edges
Textures
Patterns
Shapes
```

### ReLU

ReLU introduces non-linearity into the network:

```text
ReLU(x) = max(0, x)
```

### Max Pooling

Max pooling reduces the spatial dimensions of feature maps while retaining important features.

### Flattening

The convolutional feature maps are flattened before being passed to the fully connected classifier.

### Cross Entropy Loss

The model uses:

```python
nn.CrossEntropyLoss()
```

to measure the difference between predicted class logits and the actual class labels.

### Backpropagation

The loss is propagated backward through the network:

```python
loss.backward()
```

### Gradient Descent

The SGD optimizer updates model parameters:

```python
optimizer.step()
```

### Softmax

Softmax converts model logits into class probabilities:

```python
torch.softmax(predictions, dim=1)
```

---

# 🗂️ Module Responsibilities

| File               | Responsibility                          |
| ------------------ | --------------------------------------- |
| `data.py`          | Downloads and prepares the dataset      |
| `data_setup.py`    | Creates datasets and DataLoaders        |
| `model_builder.py` | Defines the TinyVGG CNN                 |
| `engine.py`        | Contains training and testing functions |
| `utils.py`         | Saves trained model parameters          |
| `custom_image.py`  | Downloads and predicts custom images    |
| `train.py`         | Main training and prediction pipeline   |

---

# 🔮 Future Improvements

Possible improvements for this project include:

* [ ] Train for more epochs
* [ ] Experiment with different learning rates
* [ ] Add validation split
* [ ] Add confusion matrix
* [ ] Add precision, recall, and F1-score
* [ ] Improve custom image preprocessing
* [ ] Add a dedicated inference script
* [ ] Load saved `.pth` models for inference
* [ ] Add model checkpoint loading
* [ ] Compare TinyVGG with transfer-learning models
* [ ] Experiment with ResNet or EfficientNet
* [ ] Add a web interface for predictions
* [ ] Deploy the trained model as an API
* [ ] Add automated experiment tracking

---

# 🎯 Learning Outcomes

Through this project, I gained practical experience with:

* Building CNNs with PyTorch
* Understanding convolutional layers
* Working with image tensors
* Using torchvision datasets
* Creating PyTorch DataLoaders
* Applying image transformations
* Data augmentation
* Training neural networks
* Backpropagation and optimization
* Evaluating classification models
* Saving PyTorch models
* Running inference on custom images
* Structuring a machine-learning project into reusable modules

---

# ⚠️ Notes

The current implementation contains a machine-specific Windows path in `custom_image.py`:

```python
C:\Users\susha\OneDrive\Apps\ScriptMode\data
```

For better portability when publishing the repository, this should ideally be replaced with a relative/project-based path.

For example:

```python
Path("data") / url.split("/")[-1]
```

This allows the project to run on other computers without modifying the user's local directory.

---

# 📚 References

This project was developed while learning from the **PyTorch Deep Learning** course and related resources by Daniel Bourke.

* [PyTorch Documentation](https://pytorch.org/docs/)
* [Torchvision Documentation](https://pytorch.org/vision/stable/)
* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [PyTorch Deep Learning Course](https://github.com/mrdbourke/pytorch-deep-learning)

---

# 👨‍💻 Author

**Sushanth Vaddi**

Computer Science Student | Aspiring Machine Learning Engineer

This project was created as part of my hands-on learning journey in **Deep Learning, Computer Vision, and PyTorch**.
