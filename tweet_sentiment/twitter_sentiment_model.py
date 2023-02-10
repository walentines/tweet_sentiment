# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train_preprocessed.csv')

# Cleaning the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')
corpus = []
for i in range(0, 27485):
    review = re.sub('[^a-zA-Z]', ' ', dataset['selected_text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
"""   
# Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
y = dataset.iloc[:, -1].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
"""
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Fitting the Naive Bayes to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying kFold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train, cv = 10)

# Applying Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 25, 50, 100, 150, 200], 'criterion': ['gini']},
              {'n_estimators': [10, 25, 50, 100, 150, 200], 'criterion': ['entropy']}]
grid_search = GridSearchCV(estimator = classifier,
                           scoring = 'accuracy',
                           param_grid = parameters,
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)