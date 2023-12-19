from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("data/diabetes_pre.csv")
train_data.head()


# Nominal encoding for gender (manually using a python dictionary)
# female = 0, male = 1

gender_dict = {'Female': 0, 'Male': 1}

train_data['gender'] = train_data.gender.map(gender_dict)


train_data.head()


# checking for missing values
train_data.isnull().sum()


imputer = SimpleImputer(strategy='mean')

train_data = imputer.fit_transform(train_data)


# scaling the data for higher accuracy

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)


train_data = pd.DataFrame(train_data, columns=['gender', 'age', 'hypertension', 'heart_disease',
                          'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'])


# spliiting the data into training and test data

X = train_data[train_data.columns[0:8]]
y = train_data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# training the data using Decision Trees

clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

clf.fit(X_train, y_train)


# predicting the values on test data
y_preds = clf.predict(X_test)

# print(confusion_matrix(y_test, y_preds))
cm = confusion_matrix(y_test, y_preds)

print('\nConfusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0, 0])

print('\nTrue Negatives(TN) = ', cm[1, 1])

print('\nFalse Positives(FP) = ', cm[0, 1])

print('\nFalse Negatives(FN) = ', cm[1, 0])

# create a confusion matrix
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])

# create a figure and an axes object
plt.figure(figsize=(8, 6))
ax = plt.subplot()

# plot the heatmap with annotations, a title, and labels
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()
total = cm[0, 0]+cm[0, 1]+cm[1, 0]+cm[1, 1]
accuracy = (cm[0, 0]+cm[1, 1])/total
print("\n\naccuracy is: ", accuracy)

errorRate = (cm[0, 1]+cm[1, 0])/total
print("\nError Rate is: ", errorRate)

sensitivity = cm[0, 0]/(cm[0, 0]+cm[1, 0])
print("\nSensitivity is: ", sensitivity)

specificity = cm[1, 1]/(cm[1, 1]+cm[0, 1])
print("\nSpecificity is: ", specificity)

precision = cm[0, 0]/(cm[0, 0]+cm[0, 1])
print("\nPrecision is: ", precision)

recall = cm[0, 0]/(cm[0, 0]+cm[1, 0])
print("\nRecall is: ", recall)

f_measure = (2*precision*recall)/(precision+recall)
print(f'\nF-Measure is: {f_measure}')

# visualizing the decision tree

tree.plot_tree(clf.fit(X_train, y_train))
plt.show()

