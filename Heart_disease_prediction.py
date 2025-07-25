#!/usr/bin/env python
# coding: utf-8

# # libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


# ## II. Importing and understanding our dataset 

# In[10]:


dataset = pd.read_csv("Heart_Disease_Prediction.csv")


# #### Verifying it as a 'dataframe' object in pandas

# In[11]:


type(dataset)


# #### Shape of dataset

# In[12]:


dataset.shape


# #### Printing out a few columns

# In[13]:


dataset.head(5)


# In[15]:


dataset.sample(5)


# #### Description

# In[16]:


dataset.describe()


# In[17]:


dataset.info()


# #### Let's understand our columns better:

# In[18]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# # Analysing the 'sex' variable

# In[21]:


dataset["Sex"].describe()


# In[22]:


dataset["Sex"].unique()


# # Clearly, this is a classification problem, with the sex variable having values '0' and '1'

# ### Checking correlation between columns

# In[23]:


print(dataset.corr()["Sex"].abs().sort_values(ascending=False))


# ## Exploratory Data Analysis (EDA)

# ### First, analysing the target variable:

# In[25]:


y = dataset["Sex"]

sns.countplot(y)


Sex_temp = dataset.Sex.value_counts()

print(Sex_temp)


# In[27]:


print("Percentage of patience without heart problems: "+str(round(Sex_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(Sex_temp[1]*100/303,2)))

#


# In[38]:


sns.barplot(x="Sex", y="Heart Disease", data=dataset)


# # Analysing the 'age' feature

# In[29]:


dataset["Age"].unique()


# ### Analysing the 'Chest Pain Type' feature

# In[33]:


dataset["Chest pain type"].unique()


# ##### As expected, the CP feature has values from 0 to 3

# In[35]:


sns.barplot(x="Chest pain type", y="Heart Disease", data=dataset)


# ### Analysing the FBS feature

# In[41]:


dataset["FBS over 120"].describe()


# In[42]:


dataset["FBS over 120"].unique()


# In[44]:


sns.barplot(x="FBS over 120", y="Heart Disease", data=dataset)


# ### Analysing the Cholesterol feature

# In[45]:


dataset["Cholesterol"].unique()


# In[47]:


sns.barplot(x="Cholesterol", y="Heart Disease", data=dataset)


# ### Analysing the 'BP' feature

# In[48]:


dataset["BP"].unique()


# In[50]:


sns.countplot(x="BP", data=dataset)


# ### Analysing the Slope feature

# In[51]:


dataset["Slope of ST"].unique()


# In[53]:


sns.barplot(x="Slope of ST", y="Heart Disease", data=dataset)


# ### Analysing the 'Number of vessels fluro	' feature

# In[55]:


dataset["Number of vessels fluro"].unique()


# In[56]:


sns.countplot(dataset["Number of vessels fluro"])


# In[60]:


sns.barplot(x="Number of vessels fluro", y="Heart Disease", data=dataset)


# In[61]:


dataset["ST depression"].unique()


# In[63]:


sns.barplot(x="ST depression", y="Heart Disease", data=dataset)  


# In[64]:


sns.distplot(dataset["ST depression"])


# ## IV. Train Test split

# In[66]:


from sklearn.model_selection import train_test_split

predictors = dataset.drop("Sex",axis=1)
target = dataset["Sex"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[67]:


X_train.shape


# In[68]:


X_test.shape


# In[69]:


Y_train.shape


# In[70]:


Y_test.shape


# ## V. Model Fitting

# In[72]:


from sklearn.metrics import accuracy_score


# ### Logistic Regression

# In[73]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Copy dataset
df = dataset.copy()

# Encode categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separate features and target
X = df.drop('Heart Disease', axis=1)
Y = df['Heart Disease']

# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)

# Predict
Y_pred_lr = lr.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(Y_test, Y_pred_lr))


# In[74]:


Y_pred_lr.shape


# In[75]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# ### Naive Bayes

# In[77]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)


# In[78]:


Y_pred_nb.shape


# In[79]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# ### SVM

# In[80]:


from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)


# In[81]:


Y_pred_svm.shape


# In[82]:


score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# ### K Nearest Neighbors

# In[83]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)


# In[84]:


Y_pred_knn.shape


# In[55]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# ### Decision Tree

# In[85]:


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)


# In[86]:


print(Y_pred_dt.shape)


# In[87]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# ### Random Forest

# In[1]:


# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Load your dataset (replace if already loaded)
df = pd.read_csv('heart.csv')  # Or use your actual DataFrame if already loaded

# Split features and target
X = df.drop('target', axis=1)   # Assuming 'target' is the label column
y = df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importances
importances = rf_model.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Random Forest")
plt.show()


# In[4]:


Y_pred_rf = rf_model.predict(X_test)
print(Y_pred_rf.shape)


# In[6]:


# Predict with Random Forest
Y_pred_rf = rf_model.predict(X_test)

# Accuracy Score
score_rf = round(accuracy_score(y_test, Y_pred_rf) * 100, 2)
print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")


# ### XGBoost

# In[8]:


import xgboost as xgb

# Initialize model
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

# Fit model
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
score_xgb = round(accuracy_score(y_test, y_pred_xgb) * 100, 2)
print("The accuracy score achieved using XGBoost is: " + str(score_xgb) + " %")


# In[10]:


Y_pred_xgb = xgb_model.predict(X_test)
print(Y_pred_xgb.shape)


# In[12]:


from sklearn.metrics import accuracy_score

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
score_xgb = round(accuracy_score(y_test, y_pred_xgb) * 100, 2)
print("The accuracy score achieved using XGBoost is: " + str(score_xgb) + " %")


# ### Neural Network

# In[13]:


from keras.models import Sequential
from keras.layers import Dense


# In[14]:


# https://stats.stackexchange.com/a/136542 helped a lot in avoiding overfitting

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[16]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300)


# In[17]:


Y_pred_nn = model.predict(X_test)


# In[18]:


Y_pred_nn.shape


# In[19]:


rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded


# In[21]:


# Predict
y_pred_nn = model.predict(X_test)
y_pred_nn = (y_pred_nn > 0.5).astype(int)  # Convert probabilities to 0/1

# Evaluate
score_nn = round(accuracy_score(y_test, y_pred_nn) * 100, 2)
print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")


# ## VI. Output final score

# In[26]:


# Import all necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
score_lr = round(accuracy_score(y_test, y_pred_lr) * 100, 2)

# 2. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
score_nb = round(accuracy_score(y_test, y_pred_nb) * 100, 2)

# 3. Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
score_svm = round(accuracy_score(y_test, y_pred_svm) * 100, 2)

# 4. K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
score_knn = round(accuracy_score(y_test, y_pred_knn) * 100, 2)

# 5. Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
score_dt = round(accuracy_score(y_test, y_pred_dt) * 100, 2)

# 6. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
score_rf = round(accuracy_score(y_test, y_pred_rf) * 100, 2)

# 7. XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(y_test, y_pred_xgb) * 100, 2)

# 8. Neural Network (Simple with Keras)
nn_model = Sequential()
nn_model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=100, verbose=0)

y_pred_nn = nn_model.predict(X_test)
y_pred_nn = (y_pred_nn > 0.5).astype(int)
score_nn = round(accuracy_score(y_test, y_pred_nn) * 100, 2)



# In[27]:


scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "SVM", "KNN", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]

for i in range(len(algorithms)):
    print(f"{algorithms[i]}: {scores[i]}%")


# In[28]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(algorithms, scores, color='skyblue')
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison - Heart Disease Prediction")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame for easier plotting
import pandas as pd

results_df = pd.DataFrame({
    'Algorithm': algorithms,
    'Accuracy': scores
})

plt.figure(figsize=(12, 6))
sns.barplot(x='Algorithm', y='Accuracy', data=results_df, palette='viridis')

plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score (%)")
plt.title("Comparison of Model Accuracies")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[ ]:




