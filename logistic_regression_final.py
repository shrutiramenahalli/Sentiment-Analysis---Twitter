import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def plot_roc_curve(fpr, tpr, title, num, curve):
    plt.figure(num)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + str(title))
    plt.legend([curve])
    plt.show()
    
df = pd.read_csv('training_data_final.csv', header = None)
x = df.iloc[:, 0] #first column
y = df.iloc[:,1] #second column

xtrain_1=[]
tokenizer = CountVectorizer().build_tokenizer()
lemmatizer = nltk.WordNetLemmatizer()
for X in x:
    Y=str(X).replace('\n','')
    X=tokenizer(Y)
    X=WhitespaceTokenizer().tokenize(str(X))
    X=word_tokenize(str(X))
    lemmas = [lemmatizer.lemmatize(token) for token in X]
    xtrain_1.append(str(lemmas))
    
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.2)
X = vectorizer.fit_transform(xtrain_1)

from sklearn import metrics

kf = KFold(n_splits = 5)
accuracy = []
C_range = [0.001, 0.01, 0.1, 1, 10, 100]
for Ci in C_range:
    temp = []   
    temp_pre = []
    model = LogisticRegression(penalty = 'l2', C = Ci)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        temp.append(metrics.accuracy_score(y[test], y_pred))
    accuracy.append(np.array(temp).mean())
   
plt.figure(1)
plt.errorbar(C_range,accuracy)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('C vs Accuracy')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = LogisticRegression(penalty = 'l2', C = 10)
model.fit(X_train, y_train)
preds = model.predict(X_test)

probs = model.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr, '(Logistic Regression)', 2, 'Logistic Regression')

cm = confusion_matrix(y_test, preds) 
print(classification_report(y_test, preds))
print("Accuracy:",metrics.accuracy_score(y_test, preds))
print("Precision:",metrics.precision_score(y_test, preds))
print("Recall:",metrics.recall_score(y_test, preds))

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
plot_roc_curve(fpr, tpr, '(Baseline)', 3, 'Baseline')
cm_baseline = confusion_matrix(y_test, y_pred) 
print(classification_report(y_test, y_pred))
print("Accuracy Baseline:",metrics.accuracy_score(y_test, y_pred))
print("Precision Baseline:",metrics.precision_score(y_test, y_pred))
print("Recall Baseline:",metrics.recall_score(y_test, y_pred))