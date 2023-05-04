import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
x_train, x_test, y_train, y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)
dt = DecisionTreeClassifier(max_depth=2, random_state=0)
dt.fit(x_train, y_train)
dt.predict(x_test)
tree.plot_tree(dt)
fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn = ['setosa', 'versicolor', 'virginica']
fig,axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
tree.plot_tree(dt, feature_names=fn, class_names=cn, filled=True)
fig.savefig('imagename.png')
y_pred = dt.predict(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

'''
[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0 2 1 0 1 2 1 0
 2]
[[13  0  0]
 [ 0 15  1]
 [ 0  3  6]]
 
 diagram
 '''
