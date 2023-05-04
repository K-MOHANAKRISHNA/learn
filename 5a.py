import pandas as pd
data = {"A": ["TeamA", "TeamB", "TeamB", "TeamC", "TeamA"], "B": [
 50, 40, 40, 30, 50], "C": [True, False, False, False, True]}
df = pd.DataFrame(data)
print(df)
display(df.drop_duplicates())

'''
       A   B      C
0  TeamA  50   True
1  TeamB  40  False
2  TeamB  40  False
3  TeamC  30  False
4  TeamA  50   True
A	B	C
0	TeamA	50	True
1	TeamB	40	False
3	TeamC	30	False
'''



# cross validation

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
x, y = datasets.load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=0)
k_folds = KFold(n_splits=5)
Scores = cross_val_score(clf, x, y, cv=k_folds)
print("cross validation scores:", Scores)
print("Average cv scores :", Scores.mean())
print("Number of cv scores used in Average", len(Scores))

'''

cross validation scores: [1.         0.96666667 0.83333333 0.93333333 0.8       ]
Average cv scores : 0.9066666666666666
Number of cv scores used in Average 5

'''

# Bias Variance

from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
x, y = iris_data()
x_train, x_test, y_train, y_test = train_test_split(
 x, y, test_size=0.3, random_state=123, shuffle=True, stratify=y)
tree = DecisionTreeClassifier(random_state=123)
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
 tree, x_train, y_train, x_test, y_test, loss='0-1_loss', random_seed=123, num_rounds=1000)
print(f'Average Expected loss:{round(avg_expected_loss,4)}n')
print(f'Average Bias:{round(avg_bias,4)}n')
print(f'Average Variance:{round(avg_var,4)}n')


'''
Average Expected loss:0.0607n
Average Bias:0.0222n
Average Variance:0.0393n
''''
