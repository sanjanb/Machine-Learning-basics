# 🚀 Feature Selection Techniques: RFE & SFS

## 📌 Introduction
Feature selection is a crucial step in **machine learning** that helps improve model performance by eliminating irrelevant or redundant features. This repository demonstrates **Recursive Feature Elimination (RFE)** and **Sequential Feature Selection (SFS)** using **Scikit-learn** and **MLXtend** on the **Heart Disease Dataset** from Kaggle.

## 📂 Dataset
The dataset is downloaded from Kaggle using `kagglehub`, and it contains various health parameters related to heart disease. The goal is to select the most relevant features that contribute to the target variable.

---
## 📜 Step-by-Step Code Explanation

### **1️. Import Necessary Libraries**
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import kagglehub
```
These libraries are used for:
✅ **Data handling** (`pandas`, `numpy`)
✅ **Feature selection** (`RFE`, `SequentialFeatureSelector`)
✅ **Modeling** (`LogisticRegression`, `RandomForestClassifier`)
✅ **Data scaling** (`StandardScaler`)
✅ **Dataset retrieval** (`kagglehub`)

### ** 2. Download and Load Dataset**
```python
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)

for filename in os.listdir(path):
    if filename.endswith(".csv"):
        csv_file_path = os.path.join(path, filename)
        break

df = pd.read_csv(csv_file_path)
```
✅ The Kaggle dataset is **automatically downloaded** using `kagglehub`.
✅ The script **searches for the CSV file** and loads it into a DataFrame.

### **3️. Data Preprocessing**
```python
X = df.drop('age', axis=1)  # Features
y = df['age']  # Target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
✅ `X` contains all features except `'age'` (which is used as the target).
✅ `StandardScaler` is applied to normalize the feature values.

---
## 🔥 **Feature Selection Techniques**

### **4️. Recursive Feature Elimination (RFE)**
```python
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X_scaled, y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
```
✅ **RFE** recursively eliminates the least important features.
✅ A **Logistic Regression** model is used to evaluate feature importance.
✅ Only **5 best features** are selected.

### **5️. Sequential Forward Feature Selection (SFS)**
```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
sfs = SFS(model, k_features=5, forward=True, verbose=2, scoring='accuracy', cv=5)
sfs = sfs.fit(df.drop('target', axis=1), df['target'])

print('Best accuracy score: %.2f' % sfs.k_score_)
print('Best subset (indices):', sfs.k_feature_idx_)
print('Best subset (corresponding names):', sfs.k_feature_names_)
```
✅ **SFS** selects features step-by-step using a **Random Forest Classifier**.
✅ **Cross-validation (cv=5)** ensures model generalization.
✅ The best 5 features are selected **based on accuracy score**.

---
##  **Comparison of RFE & SFS**
| Feature Selection Method | Algorithm Used | Selection Process | Output |
|----------------|------------------|----------------|--------|
| **RFE** | Logistic Regression | Recursively eliminates least important features | Best-ranked features |
| **SFS** | Random Forest | Adds features iteratively to improve model accuracy | Best-performing feature subset |

---
## 📌 **Conclusion**
Feature selection techniques like **RFE** and **SFS** help improve:
✅ Model **performance** by reducing overfitting 
✅ **Computational efficiency** by removing unnecessary features 
✅ **Interpretability** by keeping only relevant features 

By using **these methods**, we can build more accurate and efficient **machine learning models**! 🔥

---
##  **How to Use this Repo?**
1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/Feature-Selection-ML.git
```
2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run the script
```bash
python feature_selection.py
```

---
## 📢 **Contribute**
If you find this useful, feel free to **fork** and contribute! 
✅ Found a bug? Open an **Issue**!
✅ Want to improve the code? Submit a **Pull Request**!

---
##  **Star this Repo!**
If you found this helpful, don't forget to **star** ⭐ the repository!

Happy Coding! 🎯🚀

