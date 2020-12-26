import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn import metrics

def plot_roc_curve(fpr, tpr, title, num):
    plt.figure(num)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + str(title))
    plt.show()
    
df = pd.read_csv('training_data_final.csv', header = None)
x = df.iloc[:, 0] #first column
y = df.iloc[:,1] #second column

xtrain_1=[]
tokenizer = CountVectorizer().build_tokenizer()
stemmer = PorterStemmer()
for X in x:
    Y=str(X).replace('\n','')
    X=tokenizer(Y)
    X=WhitespaceTokenizer().tokenize(str(X))
    X=word_tokenize(str(X))
    stems = [stemmer.stem(token) for token in X]
    xtrain_1.append(str(stems))
    
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.2)
X = vectorizer.fit_transform(xtrain_1)

kf = KFold(n_splits = 10)
mean_error = []
std_error = []
temp = []   
model = BernoulliNB()
for train, test in kf.split(X):
    model.fit(X[train], y[train])
    preds = model.predict(X[test])
    temp.append(mean_squared_error(y[test], preds)) 
mean_error.append(np.array(temp).mean())
std_error.append(np.array(temp).std())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = BernoulliNB()
model.fit(X_train, y_train)
preds = model.predict(X_test)

probs = model.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr, '(Naive Bayes)', 1)

cm = confusion_matrix(y_test, preds) 
print(classification_report(y_test, preds))
print("Accuracy Baseline:",metrics.accuracy_score(y_test, preds))
print("Precision Baseline:",metrics.precision_score(y_test, preds))
print("Recall Baseline:",metrics.recall_score(y_test, preds))

from sklearn.dummy import DummyClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test)

probs = dummy_clf.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr, '(Baseline)', 2)
cm_baseline = confusion_matrix(y_test, y_pred) 
print(classification_report(y_test, y_pred))
print("Accuracy Baseline:",metrics.accuracy_score(y_test, y_pred))
print("Precision Baseline:",metrics.precision_score(y_test, y_pred))
print("Recall Baseline:",metrics.recall_score(y_test, y_pred))

