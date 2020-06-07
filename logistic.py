import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import warnings

warnings.default_action = "ignore"

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=21)

lm = LogisticRegression()

lm.fit(X_train, y_train)

print("The training accuracy is ", lm.score(X_train, y_train))

print("The testing accuracy is ", lm.score(X_test, y_test))

y_pred = lm.predict(X_test)

print("The accuracy score is ", accuracy_score(y_test, y_pred))


######################################################################################################

from sklearn.neighbors import KNeighborsClassifier  

no = 5

knn = KNeighborsClassifier(n_neighbors=no)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN accuracy score is ", accuracy_score(y_test, y_pred))