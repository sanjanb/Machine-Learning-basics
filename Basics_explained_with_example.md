# Heart Disease Dataset Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Pandas Version](https://img.shields.io/badge/pandas-1.4.3%2B-orange.svg)](https://pandas.pydata.org/)
[![KaggleHub Version](https://img.shields.io/badge/kagglehub-0.1.0%2B-green.svg)](https://pypi.org/project/kagglehub/)

This repository contains an analysis of the Heart Disease Dataset using Python and Pandas. The dataset is downloaded using the `kagglehub` library, and basic exploratory data analysis (EDA) is performed to understand the data structure and contents.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Data Exploration](#data-exploration)
- [Basics of Machine Learning](#basics-of-machine-learning)
- [License](#license)
- [Author](#author)

## Introduction

The Heart Disease Dataset is a collection of medical data used to predict the presence of heart disease in patients. This repository demonstrates how to download and analyze the dataset using Python and Pandas, and provides a brief overview of the basics of machine learning.

## Dataset

The dataset used in this analysis is available on Kaggle and can be downloaded using the `kagglehub` library. The dataset contains various medical attributes of patients and a target variable indicating the presence of heart disease.

## Environment Setup

Before running the analysis, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install pandas kagglehub
```

## Data Exploration

The following code demonstrates how to download the dataset and perform basic EDA using Pandas.

```python
import pandas as pd
import os
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)

# Load the data from the CSV file
df = pd.read_csv(path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display the last few rows of the dataset
print("\nLast few rows of the dataset:")
print(df.tail())

# Get a summary of the dataset
print("\nDataset summary:")
print(df.info())

# Get statistical summary of the dataset
print("\nStatistical summary of the dataset:")
print(df.describe())

# Check for missing values in the dataset
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Check the number of unique values in each column
print("\nNumber of unique values in each column:")
print(df.nunique())

# List the column names
print("\nColumn names:")
print(df.columns)

# Get the shape of the dataset
print("\nShape of the dataset (rows, columns):")
print(df.shape)

# Check for duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())
```

### Output Explanation

- **`df.head()`**: Displays the first few rows of the dataset.
- **`df.tail()`**: Displays the last few rows of the dataset.
- **`df.info()`**: Provides a summary of the dataset, including data types and non-null counts.
- **`df.describe()`**: Provides statistical summary of numerical columns.
- **`df.isnull().sum()`**: Checks for missing values in each column.
- **`df.nunique()`**: Counts the number of unique values in each column.
- **`df.columns`**: Lists the column names in the dataset.
- **`df.shape`**: Returns the dimensions of the dataset (number of rows and columns).
- **`df.duplicated().sum()`**: Checks for duplicate rows in the dataset.

## Basics of Machine Learning

Machine Learning (ML) is a field of artificial intelligence that focuses on building models that can learn from and make predictions on data. Here are some fundamental concepts:

### 1. Types of Machine Learning

- **Supervised Learning**: The model is trained on labeled data.
  - **Classification**: Predicting a discrete label (e.g., presence or absence of heart disease).
  - **Regression**: Predicting a continuous value (e.g., predicting blood pressure).

- **Unsupervised Learning**: The model is trained on unlabeled data.
  - **Clustering**: Grouping similar data points together.
  - **Dimensionality Reduction**: Reducing the number of features while retaining important information.

- **Reinforcement Learning**: The model learns by interacting with an environment and receiving rewards or penalties.

### 2. Machine Learning Workflow

1. **Data Collection**: Gather and prepare the dataset.
2. **Data Preprocessing**: Clean and preprocess the data.
3. **Exploratory Data Analysis (EDA)**: Understand the data through visualization and statistics.
4. **Model Selection**: Choose an appropriate machine learning algorithm.
5. **Model Training**: Train the model on the training data.
6. **Model Evaluation**: Evaluate the model's performance on the test data.
7. **Model Deployment**: Deploy the model for real-world use.

### 3. Key Libraries

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning algorithms.
- **Matplotlib/Seaborn**: For data visualization.
- **KaggleHub**: For downloading datasets from Kaggle.

### 4. Simple Example of Machine Learning Workflow

Here’s a simple example using a classification task with the Heart Disease Dataset.

#### Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
```

#### Step 2: Download and Load the Dataset

```python
# Download the latest version of the dataset
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)

# Load the data from the CSV file
df = pd.read_csv(path)
```

#### Step 3: Exploratory Data Analysis (EDA)

```python
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display the last few rows of the dataset
print("\nLast few rows of the dataset:")
print(df.tail())

# Get a summary of the dataset
print("\nDataset summary:")
print(df.info())

# Get statistical summary of the dataset
print("\nStatistical summary of the dataset:")
print(df.describe())

# Check for missing values in the dataset
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Check the number of unique values in each column
print("\nNumber of unique values in each column:")
print(df.nunique())

# List the column names
print("\nColumn names:")
print(df.columns)

# Get the shape of the dataset
print("\nShape of the dataset (rows, columns):")
print(df.shape)

# Check for duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())
```

#### Step 4: Data Preprocessing

```python
# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing values if any
df.fillna(df.mean(), inplace=True)

# Feature selection
X = df.drop('target', axis=1)  # Features
y = df['target']               # Target variable
```

#### Step 5: Split the Dataset

```python
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Step 6: Feature Scaling

```python
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Step 7: Model Selection and Training

```python
# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
```

#### Step 8: Model Evaluation

```python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: [Sanjan B M](https://github.com/sanjanb)
```

### **Steps to Add and Push the `README.md` File**

1. **Edit the `README.md` File**:
   - Open the `README.md` file in a text editor and paste the above Markdown code.

2. **Commit the Changes**:
   ```bash
   git add README.md
   git commit -m "Update README.md with detailed usage and basics of machine learning"
   ```

3. **Push the Changes**:
   ```bash
   git push origin main
   ```

