# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=0)
clf1 = DecisionTreeClassifier()
clf1.fit(x_train, y_train)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall(y_test,clf1.predict(x_test)),precision(y_test,clf1.predict(x_test)))

clf2 = GaussianNB()
clf2.fit(x_train, y_train)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(y_test,clf2.predict(x_test)),precision(y_test,clf2.predict(x_test)))

results = {
  "Naive Bayes Recall": 0.38,
  "Naive Bayes Precision": 0.58,
  "Decision Tree Recall": 0.51,
  "Decision Tree Precision": 0.61
}