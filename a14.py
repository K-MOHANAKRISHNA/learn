import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=pd.read_csv('iris.data.csv',names=['sepal_length','sepal_width','petal_length','petal_width','class'])
X_train,X_test,Y_train,Y_test=train_test_split(iris.drop('class',axis=1),iris['class'],test_size=0.3,random_state=42)
clf=svm.SVC(kernel='rbf')
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(Y_test,y_pred)
print("Accuracy:",accuracy)

'''

Accuracy: 1.0

'''
