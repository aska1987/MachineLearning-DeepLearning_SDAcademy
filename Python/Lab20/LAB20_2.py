from sklearn import datasets
import pandas as pd

#Load the iris dataset
iris = datasets.load_iris()

#create a data frame named df (including target and feature names)
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['class'] = iris.target

#summarize the data frame
df.shape
df.head(20)
a = df.describe()
df.groupby('class').size()

#univariate plot
#box plot
df1 = df.drop('class', axis=1)
df1.boxplot()
df1.plot(kind='box', subplots=True, layout=(2,2))

#histogram
df1.hist()
df1.columns
df1['sepal length (cm)'].hist()
df1.plot(kind='hist', subplots=True, layout=(2,2))

#multivariate plot
#scatter matrix
pd.scatter_matrix(df1)

#split independent variables and depedent variable
y = df['class']
x = df.drop('class', axis=1)

#split dataset
from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,
                                                                    test_size=.2, random_state=1)
from sklearn.linear_model import LinearRegression
lm_reg = LinearRegression()
lm_reg.fit(x_train, y_train)
pred = lm_reg.predict(x_test)
pred

from sklearn.metrics import r2_score, accuracy_score
r2_score(y_train, lm_reg.predict(x_train))

from sklearn.model_selection import cross_val_score
cross_val_score(lm_reg, x_train, y_train, cv=5, scoring='r2')

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtree_class = DecisionTreeClassifier(random_state=0)
dtree_class.fit(x_train, y_train)
pred = dtree_class.predict(x_test)
pred

#model evaluation
r2_score(y_train, dtree_class.predict(x_train))
accuracy_score(y_train, dtree_class.predict(x_train))
cross_val_score(dtree_class, x_train, y_train, cv=5, scoring='r2')

#tree visualization
from IPython.display import Image
from sklearn import tree
import pydotplus

dot_data = tree.export_graphviz(dtree_class, out_file=None, 
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

dtree_class = DecisionTreeClassifier(random_state=0, max_depth=3)
dtree_class.fit(x_train, y_train)
x_train.columns
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
dot_data = tree.export_graphviz(dtree_class, out_file=None, 
                                feature_names=feature_names,
                                rounded=True, filled=True,
                                class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_pdf('dtree.pdf')
graph.write_png('dtree.png')

#random forest
from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators=100, max_depth=5)
rf_class.fit(x_train, y_train)
pred = rf_class.predict(x_test)
pred

#model evaluation
r2_score(y_train, rf_class.predict(x_train))
accuracy_score(y_train, rf_class.predict(x_train))
cross_val_score(rf_class, x_train, y_train, cv=5, scoring='r2')

#pima indians diabetes example
pima = pd.read_csv("pima-indians-diabetes.csv", header=None)
pima.columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

pima.shape
pima.head(20)
pima.describe()
pima.plot(kind='box', subplots=True, layout=(3,3))
pima.plot(kind='hist', subplots=True, layout=(3,3))
pd.scatter_matrix(pima)

x = pima.drop('class', axis=1)
y = pima['class']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=.2, random_state=7)

lm_reg = LinearRegression()
lm_reg.fit(x_train, y_train)
lm_reg.predict(x_test)

#evaluation
r2_score(y_train,lm_reg.predict(x_train))
cross_val_score(lm_reg, x, y, cv=5, scoring='r2')

#decision tree
dtree_class = DecisionTreeClassifier(max_depth=3, random_state=0)
dtree_class.fit(x_train, y_train)
dtree_class.predict(x_test)

r2_score(y_train,dtree_class.predict(x_train))
accuracy_score(y_train,dtree_class.predict(x_train))

dot_data = tree.export_graphviz(dtree_class, out_file=None, 
                                feature_names=pima.columns[:-1],
                                rounded=True, filled=True,
                                class_names=['negative', 'positive'])
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

#random forest
from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators=100, max_depth=5)
rf_class.fit(x_train, y_train)
pred = rf_class.predict(x_test)
pred
r2_score(y_train,rf_class.predict(x_train))
accuracy_score(y_train,rf_class.predict(x_train))
cross_val_score(rf_class, x_train, y_train, cv=5, scoring='r2')
