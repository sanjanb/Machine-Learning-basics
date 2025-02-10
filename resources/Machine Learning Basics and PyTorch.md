# Machine Learning Basics and PyTorch

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.12%2B-green.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is designed to help beginners understand the fundamentals of machine learning, including data exploration, ethics, types of bias, sources of bias, examples of bias, ethical issues in AI/ML, and the importance of AI/ML frameworks. The repository uses the Heart Disease Dataset for practical examples and includes a simple PyTorch example to demonstrate neural network training.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [PyTorch](#pytorch)
  - [Overview](#overview)
  - [Dynamic Computation Graph](#dynamic-computation-graph)
  - [Ease of Use](#ease-of-use)
  - [Optimizer: Adam](#optimizer-adam)
  - [Example Code](#example-code)
- [AI/ML Frameworks](#ai/ml-frameworks)
- [Importance of AI/ML Frameworks](#importance-of-ai/ml-frameworks)
- [License](#license)
- [Author](#author)

## Introduction

Machine Learning (ML) is a field of artificial intelligence that focuses on building models that can learn from and make predictions on data. This repository provides a beginner-friendly guide to understanding the basics of machine learning, including data exploration, ethics, and practical examples using PyTorch.

## Dataset

The dataset used in this analysis is the Heart Disease Dataset, available on Kaggle. The dataset contains various medical attributes of patients and a target variable indicating the presence of heart disease.

## Environment Setup

Before running the analysis, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install pandas kagglehub scikit-learn matplotlib seaborn torch torchvision
```



## PyTorch

### Overview

PyTorch is an open-source machine learning framework developed by Facebook's AI Research lab. It is widely used in research and production due to its dynamic computation graph, ease of use, and flexibility.

### Dynamic Computation Graph

PyTorch uses a dynamic computation graph, which means the graph is built on-the-fly during runtime. This makes it easier to debug and experiment with different model architectures.

### Ease of Use

PyTorch is known for its simplicity and ease of use, making it accessible to both beginners and experienced developers. Its syntax is similar to NumPy, which simplifies the transition from traditional Python programming.

### Optimizer: Adam

The Adam optimizer is one of the most commonly used optimizers in PyTorch. It combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. Adam computes adaptive learning rates for each parameter, which helps in faster convergence.

### Example Code

Below is a simple example of training a neural network using PyTorch with the Fashion MNIST dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define transformations for dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load dataset (Fashion MNIST)
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

print("Training complete!")

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")

# Visualize some test images and their predicted labels
dataiter = iter(testloader)
images, labels = next(dataiter)

# Print images
imshow(torchvision.utils.make_grid(images))

# Predict labels
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{predicted[j].item():5}' for j in range(4)))

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
```

### Explanation of Example Code

1. **Transformations**:
   - Convert images to tensors and normalize them.

2. **Loading Dataset**:
   - Load the Fashion MNIST dataset for training and testing.

3. **Defining the Model**:
   - Create a simple neural network with one hidden layer.

4. **Initializing Model, Loss Function, and Optimizer**:
   - Use `nn.CrossEntropyLoss` for the loss function.
   - Use `optim.Adam` for the optimizer.

5. **Training Loop**:
   - Train the model for a specified number of epochs.
   - Zero the gradients, perform a forward pass, compute the loss, perform a backward pass, and update the weights.

6. **Testing Loop**:
   - Evaluate the model on the test dataset.
   - Print the accuracy of the model.

7. **Visualization**:
   - Display some test images and their predicted labels.

## AI/ML Frameworks

AI/ML frameworks are software libraries or tools designed to build AI/ML models. They simplify the process of model development and deployment by providing pre-built algorithms, tools, and APIs.

### Importance of AI/ML Frameworks

- **Speed up Development Time**: Reduce the time required to build and train models.
- **Increase Productivity**: Provide efficient tools for data preprocessing, model training, and evaluation.
- **Accessibility**: Make AI/ML accessible to developers of all levels.
- **Continuous Process**: Support the entire lifecycle of model development, from data collection to deployment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**Author**: [Sanjan B M](https://github.com/sanjanb)



### **Additional Resources**

To further enhance your repository, you can add additional resources such as:

- **Jupyter Notebooks**: Interactive notebooks for hands-on learning.
- **Tutorials**: Step-by-step guides on specific topics.
- **Links to External Resources**: Articles, videos, and courses for deeper understanding.

Here’s an example of how you can add a section for additional resources:

## Additional Resources

- **Jupyter Notebooks**: [Notebooks](notebooks/)
- **Tutorials**: [Tutorials](tutorials/)
- **External Resources**:
  - [Coursera Machine Learning Course](https://www.coursera.org/learn/machine-learning)
  - [Fast.ai](https://www.fast.ai/)
  - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
  - [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
  - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
