# train=pd.read_csv("C:\\Users\\Dell\\Desktop\\scripts\\Machine_Learning\\Supervised_Model\\Logistic_Regression\\Cancer_Prediction\\data.csv")

import pandas as pd
train=pd.read_csv("C:\\Users\\Dell\\Desktop\\scripts\\Machine_Learning\\Supervised_Model\\Logistic_Regression\\Cancer_Prediction\\data.csv")
train.info()
print(train)
train.shape

#checking out null values
train.isnull().sum()
train.duplicated().sum()

train['diagnosis'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sb
sb.countplot(x='diagnosis', data=train)
train.columns
col=['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
sb.pairplot(hue='diagnosis', data=train[col])

cor=train.corr().round(2)
import numpy as np
mask=np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
# set figure size
f,ax=plt.subplots(figsize=(20,20))
cmap=sb.diverging_palette(220,10,as_cmap=True)
sb.heatmap(cor, mask=mask,cmap=cmap, vmin=-1, vmax=1,center=0
       ,square=True,linewidths=0.5,cbar_kws={'shrink': 0.5},annot=True)
plt.tight_layout()

train.columns
a=['radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
train=train.drop(a,axis=1)
train.columns

# draw heatmap again
"""
cor=train.corr().round(2)
import numpy as np
mask=np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
# set figure size
f,ax=plt.subplots(figsize=(20,20))
cmap=sb.diverging_palette(220,10,as_cmap=True)
sb.heatmap(cor, mask=mask,cmap=cmap, vmin=-1, vmax=1,center=0
       ,square=True,linewidths=0.5,cbar_kws={'shrink': 0.5},annot=True)
plt.tight_layout()
"""

# check heatmap
train.drop(['perimeter_mean','area_mean','perimeter_se','area_se'],
           inplace=True, axis=1)
train.drop(['concavity_mean','concave points_mean','concavity_se',
            'concave points_se'],inplace=True, axis=1)
train.columns
train_data=train.drop(["diagnosis","id"], axis=1)
target_data=train["diagnosis"]

target_data.shape
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(train_data,target_data,test_size=0.2, random_state=2)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

y=model.fit(X_train,Y_train)
predicted_t=model.predict(X_test)
model.predict([[428,1.13,16.62,4,5,2,5,6,3,5,0.02009, 0.002377]])              # testing by own data, give all rows values

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(predicted_t,Y_test))      # 0.92

print(confusion_matrix(predicted_t,Y_test))
print(classification_report(predicted_t,Y_test))

# check accuracy on test data
y=model.fit(X_test,Y_test)
predicted=model.predict(X_train)
print(accuracy_score(predicted,Y_train))     #0.87
