# PyTorch Basics and Regression Model 🚀

This repository provides a beginner-friendly introduction to **PyTorch**, covering tensor operations, automatic differentiation, and building a simple regression model using PyTorch's `nn.Module` framework.

## 📌 Topics Covered
1. **Tensor Basics** – Creating and manipulating tensors
2. **Tensor Operations** – Addition and matrix multiplication
3. **Tensor Attributes** – Checking shape and data types
4. **PyTorch Autograd** – Automatic differentiation
5. **Building a Regression Model** – Training a simple linear regression model using synthetic data

---

## 🔹 1. Tensor Basics 🧮
PyTorch tensors are similar to NumPy arrays but support GPU acceleration. This section demonstrates how to create **scalars, vectors, matrices, and tensors** using PyTorch.

```python
import torch

# Tensor creation
scalar = torch.tensor(7)
vector = torch.tensor([1,2,3])
matrix = torch.tensor([[1,2,3],[4,5,6]])
tensor = torch.rand((3,3))

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)
print("Tensor:", tensor)
```

---

## 🔹 2. Tensor Operations ⚡
Basic operations such as **addition** and **matrix multiplication** are fundamental in deep learning.

```python
# Tensor operations
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[5,6],[7,8]])
print("Addition:\n",a+b)
print("Matrix multiplication:\n", torch.matmul(a,b))
```

---

## 🔹 3. Tensor Attributes 📏
Tensors have key attributes like **shape** and **data type**.

```python
print("Shape:", tensor.shape)
print("Data type:", tensor.dtype)
```

---

## 🔹 4. PyTorch Autograd ✨
PyTorch provides automatic differentiation using **autograd**. This section demonstrates how to compute gradients.

```python
# Automatic differentiation
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x**2 + 1

# Compute gradients
y.backward()
print("Gradient of y with respect to x:", x.grad)
```

### 🔹 Gradient Accumulation
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
y = x**2 + 3
z = y.sum()
z.backward()
print("Gradients:", x.grad)
```

---

## 🔹 5. Building a Regression Model 📈
This section demonstrates how to create and train a **linear regression model** using PyTorch's `nn.Module` framework.

### 🛠️ Data Preparation
```python
import numpy as np

# Generate synthetic data
x = np.random.rand(100,1).astype(np.float32)
X = torch.from_numpy(x)
y = 3* X + 7 + torch.from_numpy(np.random.normal(0, 0.1, (100, 1)).astype(np.float32))
Y = y
```

### 🏗️ Define the Model
```python
import torch

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
```

### 🎯 Training the Model
```python
# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### 📊 Visualizing Results
```python
import matplotlib.pyplot as plt

with torch.no_grad():
    plt.scatter(x, y.numpy(), label='Data')
    plt.plot(x, model(X).detach().numpy(), label='Regression')
    plt.legend()
    plt.show()
```

---

## 📌 Summary
This tutorial covers the basics of **PyTorch tensors, autograd, and linear regression modeling**. This is an excellent starting point for beginners who want to explore **deep learning and AI** using PyTorch.

🚀 **Next Steps:** Try implementing **a classification model** or **train on a real-world dataset**!

📢 **Contribute:** Found a bug or have a suggestion? Open a PR or issue!

---

🔗 **References & Further Reading:**
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- Deep Learning with PyTorch: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

