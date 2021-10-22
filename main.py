import pandas as pd

#data-preproc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#pipeline-builder
from sklearn.pipeline import Pipeline

#metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('./resources/Womens Clothing E-Commerce Reviews.csv')

# print(df.isnull().sum())
print(f'df.shape before dropping na:\t{df.shape}\n')

df.dropna(inplace=True)
print(f'df.shape after dropping na:\t{df.shape}\n')

# print(f'df.columns:\n {df.columns}')

X = df['Review Text']
# print(X.head())
y = df['Recommended IND']

print(f'y values:')
for i in range(len(y.value_counts())):
    print(f'class {i}: {round(y.value_counts()[i]/len(df.index)*100)} ({y.value_counts()[i]})')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def show_accuracy_score(classifier):

    clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', classifier())])
    clf.fit(X_train, y_train)
    acc_score = clf.score(X_test, y_test)

    return f'{classifier.__name__}:\n{acc_score}\n{confusion_matrix(y_test,clf.predict(X_test))}\n{classification_report(y_test,clf.predict(X_test))}\n'


classifiers = [MultinomialNB, LinearSVC, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier]

for classifier in classifiers:
    print(show_accuracy_score(classifier))

# winner:
# LinearSVC:
# 0.8924333487440284