Heart Disease Prediction Project
This project aims to predict the likelihood of heart disease in patients using various medical attributes and machine learning techniques.

📌 Project Overview
Heart disease is one of the leading causes of death globally. Early prediction can help in taking timely preventive measures. This project utilizes a structured dataset of patient records to train a classification model that can predict the presence of heart disease.

📂 Steps Performed
1. Library Import
Imported necessary Python libraries such as:

NumPy, Pandas for data manipulation

Matplotlib, Seaborn for visualization

Warnings and OS for utility functions

python
Copy
Edit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
2. Dataset Import and Exploration
Loaded the dataset using pandas.

Performed initial inspection like:

Viewing column names

Checking for null values

Understanding data types

python
Copy
Edit
df = pd.read_csv("heart.csv")
df.info()
df.isnull().sum()
3. Data Visualization
Used Seaborn and Matplotlib to explore relationships and trends in the data:

Correlation heatmap

Histograms and bar charts

Pair plots

python
Copy
Edit
sns.heatmap(df.corr(), annot=True)
sns.countplot(x='target', data=df)
4. Feature Engineering
Checked the correlation between features and the target.

Encoded categorical variables where necessary.

Scaled features using StandardScaler.

python
Copy
Edit
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
5. Train-Test Split
The dataset was split into training and test sets using an 80-20 split.

python
Copy
Edit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
6. Model Building
Trained multiple classification models including:

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest

Support Vector Machine (SVM)

python
Copy
Edit
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
7. Model Evaluation
Evaluated models using:

Accuracy Score

Confusion Matrix

Classification Report

ROC-AUC Curve

python
Copy
Edit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred))
✅ Output
The final model provides predictions about whether a person is likely to suffer from heart disease based on medical attributes.

Models are compared based on their accuracy and performance metrics.

🛠️ Technologies Used
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

📊 Conclusion
This project demonstrates a full machine learning workflow for heart disease prediction. The notebook showcases data exploration, preprocessing, training, evaluation, and comparison of different models. The resulting model can be further enhanced using hyperparameter tuning and ensemble techniques.

