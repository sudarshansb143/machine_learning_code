# Note about the knn is that it requires data into the numerical format
# Most of the sklearn model requires data into the proper format i. e. numerical or the float

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.style.use("seaborn")
frame = pd.read_csv("titanic.csv")
x = frame
y = frame["Survived"]
x.drop(["Name", "Sex", "Survived"], axis= 1, inplace=True)
x.Age.fillna(x.Age.mean(), inplace=True)
a = pd.get_dummies(x.PClass).reset_index().drop(["*", "index"], axis = 1)
new_x = pd.concat([x, a], axis = 1)
new_x.drop("PClass", axis = 1, inplace= True)
scaler = StandardScaler()
scaled_x = scaler.fit_transform(new_x)

# Spliting the data in training and test
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, train_size=0.8, random_state=1)

# Score
score = []
# Now the modelling started
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    score.append(knn.score(X_test,y_test))

for i,j in enumerate(score):
     print(str(i+1) + " Nwighbours score is " + str(j))

neighbour = 6
final_knn = KNeighborsClassifier(n_neighbors=neighbour)
final_knn.fit(X_train,y_train)
y_pred = final_knn.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("Accuracy score is ", accuracy_score(y_test, y_pred))
"""Accuracy score is  0.8555133079847909"""