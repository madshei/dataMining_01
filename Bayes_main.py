import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
data = 'data/diabetes_pre.csv'

df = pd.read_csv(data)

categorical = [var for var in df.columns if df[var].dtype == 'O']
df[categorical].isnull().sum()
X = df.drop(['diabetes'], axis=1)

y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
X_train.shape, X_test.shape


# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
# print(categorical)
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
# print(df.columns)
# print(numerical)
X_train[numerical].isnull().mean()
X_test[numerical].isnull().sum()

encoder = ce.OneHotEncoder(
    cols=['gender'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.head()
cols = X_train.columns

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

X_train.head()


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)

GaussianNB(priors=None, var_smoothing=1e-09)
y_pred = gnb.predict(X_test)


print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


y_pred_train = gnb.predict(X_train)
print(
    'Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

cm = confusion_matrix(y_test, y_pred)

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

# show the plot
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
