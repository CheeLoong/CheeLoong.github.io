---
title: "Model Selection on Credit Card Fraud dataset"
date: 2019-09-05
permalink: /ccfraud-ms
tags: [model selection, feature selection, parameter tuning, credit card fraud]
excerpt: "Choosing the best models out of logreg, randforest, and svc"
mathjax: "true"
---


## Libraries, Modules, and Functions


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight

from pylab import rcParams
import warnings
```


```python
%matplotlib inline
rcParams['figure.figsize'] = 10, 6
warnings.filterwarnings('ignore')
sns.set(style = 'white')
```


```python
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```


```python
def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
```


```python
def generate_auc_roc_curve(clf, X_test, y_test, title):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label = 'ROC Curve with AUC =' + str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate / Recall')
    plt.title(title)
    plt.legend(loc=4)
```


```python
def generate_pr_curve(clf, X_test, y_test, title):
    y_score = clf.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test,  y_score)
    aupc = average_precision_score(y_test, y_score)
    plt.plot(recall,precision,label = 'PR Curve with AUC =' + str(aupc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc=4)
```


```python
d_path = "/content/data/"
folder = "cc-fraud"

if not os.path.exists(d_path + folder):
    os.makedirs(d_path + folder)

# path = Path(d_path + folder)
# path.mkdir(parents=True, exist_ok=True)
```


```python
df = pd.read_csv(d_path + folder + '/creditcard.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>0.090794</td>
      <td>-0.551600</td>
      <td>-0.617801</td>
      <td>-0.991390</td>
      <td>-0.311169</td>
      <td>1.468177</td>
      <td>-0.470401</td>
      <td>0.207971</td>
      <td>0.025791</td>
      <td>0.403993</td>
      <td>0.251412</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>-0.166974</td>
      <td>1.612727</td>
      <td>1.065235</td>
      <td>0.489095</td>
      <td>-0.143772</td>
      <td>0.635558</td>
      <td>0.463917</td>
      <td>-0.114805</td>
      <td>-0.183361</td>
      <td>-0.145783</td>
      <td>-0.069083</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>0.207643</td>
      <td>0.624501</td>
      <td>0.066084</td>
      <td>0.717293</td>
      <td>-0.165946</td>
      <td>2.345865</td>
      <td>-2.890083</td>
      <td>1.109969</td>
      <td>-0.121359</td>
      <td>-2.261857</td>
      <td>0.524980</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>-0.054952</td>
      <td>-0.226487</td>
      <td>0.178228</td>
      <td>0.507757</td>
      <td>-0.287924</td>
      <td>-0.631418</td>
      <td>-1.059647</td>
      <td>-0.684093</td>
      <td>1.965775</td>
      <td>-1.232622</td>
      <td>-0.208038</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>0.753074</td>
      <td>-0.822843</td>
      <td>0.538196</td>
      <td>1.345852</td>
      <td>-1.119670</td>
      <td>0.175121</td>
      <td>-0.451449</td>
      <td>-0.237033</td>
      <td>-0.038195</td>
      <td>0.803487</td>
      <td>0.408542</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (284807, 31)



## Data Preparation

### Train-test-split


```python
from sklearn.model_selection import train_test_split
```


```python
X = df.drop('Class',axis=1)
y = df['Class']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
```

## Feature Selection

Let's use a bunch of feature selection techniques to select features that are important to each of these techniques. We then ensemble the techniques by choosing features that are selected by all these techniques.

### Univariate Feature Selection (KBest)


```python
from sklearn.feature_selection import SelectKBest, f_classif
```


```python
# k can be any number of features, because we will inspect the score table to choose k
univar = SelectKBest(score_func=f_classif, k=10)
univar = univar.fit(X_train,y_train)
```


```python
feature_scores_df = pd.DataFrame({'Feature':list(X_train.columns),
                                     'Scores':univar.scores_})
```


```python
feature_scores_df.sort_values(by='Scores', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>V17</td>
      <td>26765.898608</td>
    </tr>
    <tr>
      <th>14</th>
      <td>V14</td>
      <td>23220.134963</td>
    </tr>
    <tr>
      <th>12</th>
      <td>V12</td>
      <td>16580.228510</td>
    </tr>
    <tr>
      <th>10</th>
      <td>V10</td>
      <td>11685.480474</td>
    </tr>
    <tr>
      <th>16</th>
      <td>V16</td>
      <td>9220.831983</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V3</td>
      <td>9061.218737</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V7</td>
      <td>8938.782653</td>
    </tr>
    <tr>
      <th>11</th>
      <td>V11</td>
      <td>5608.447011</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V4</td>
      <td>4250.797286</td>
    </tr>
    <tr>
      <th>18</th>
      <td>V18</td>
      <td>2912.707215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V1</td>
      <td>2514.609155</td>
    </tr>
    <tr>
      <th>9</th>
      <td>V9</td>
      <td>2296.382283</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V5</td>
      <td>2170.045228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>1940.333257</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V6</td>
      <td>460.998614</td>
    </tr>
    <tr>
      <th>19</th>
      <td>V19</td>
      <td>266.121384</td>
    </tr>
    <tr>
      <th>21</th>
      <td>V21</td>
      <td>249.444910</td>
    </tr>
    <tr>
      <th>20</th>
      <td>V20</td>
      <td>123.705394</td>
    </tr>
    <tr>
      <th>8</th>
      <td>V8</td>
      <td>74.007520</td>
    </tr>
    <tr>
      <th>27</th>
      <td>V27</td>
      <td>48.997422</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Time</td>
      <td>39.036114</td>
    </tr>
    <tr>
      <th>28</th>
      <td>V28</td>
      <td>18.456284</td>
    </tr>
    <tr>
      <th>13</th>
      <td>V13</td>
      <td>10.251747</td>
    </tr>
    <tr>
      <th>24</th>
      <td>V24</td>
      <td>9.653397</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Amount</td>
      <td>7.968086</td>
    </tr>
    <tr>
      <th>25</th>
      <td>V25</td>
      <td>4.307909</td>
    </tr>
    <tr>
      <th>26</th>
      <td>V26</td>
      <td>3.968576</td>
    </tr>
    <tr>
      <th>23</th>
      <td>V23</td>
      <td>3.435772</td>
    </tr>
    <tr>
      <th>15</th>
      <td>V15</td>
      <td>3.418834</td>
    </tr>
    <tr>
      <th>22</th>
      <td>V22</td>
      <td>2.181606</td>
    </tr>
  </tbody>
</table>
</div>




```python
selected_features = feature_scores_df[feature_scores_df['Scores'] > 1000].sort_values(by='Scores', ascending=False)
```


```python
# creating a list to save the features that we have selected using KBest
kbest_list = list(selected_features['Feature'])
```


```python
kbest_list
```




    ['V17',
     'V14',
     'V12',
     'V10',
     'V16',
     'V3',
     'V7',
     'V11',
     'V4',
     'V18',
     'V1',
     'V9',
     'V5',
     'V2']



### Recursive Feature Elimination (RFECV)


```python
from sklearn.feature_selection import RFECV
```


```python
lr = LogisticRegression()
rfc = RandomForestClassifier()
svm = SVC()
```


```python
rfecv_lr = RFECV(estimator=lr, step=1, cv=5, scoring='f1')
rfecv_lr = rfecv_lr.fit(X_train, y_train)
print('Optimal number of features :', rfecv_lr.n_features_)
print('Best features of logreg :', X_train.columns[rfecv_lr.support_])
```

    Optimal number of features : 22
    Best features of logreg : Index(['V1', 'V2', 'V4', 'V6', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
           'V16', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V26', 'V27',
           'V28'],
          dtype='object')



```python
rfecv_lr.grid_scores_
```




    array([0.56308217, 0.61697209, 0.63556189, 0.6472141 , 0.65075163,
           0.68421554, 0.69465623, 0.7091661 , 0.71124067, 0.71787594,
           0.7129905 , 0.7110632 , 0.71496645, 0.71313139, 0.71613427,
           0.71677977, 0.71093857, 0.70989027, 0.7109658 , 0.71195648,
           0.70983509, 0.71950185, 0.71573249, 0.71669095, 0.70904056,
           0.70899277, 0.70899277, 0.70899277, 0.70791558, 0.64549079])




```python
plt.figure()
plt.xlabel("Number of features selected by Logistic Regression")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv_lr.grid_scores_) + 1), rfecv_lr.grid_scores_)
plt.show()
```


![png](output_30_0.png)



```python
rfecv_lr_list = list(X_train.columns[rfecv_lr.support_])
```


```python
# warning: this code takes a while to be executed, uncomment to run
# rfecv_rfc = RFECV(estimator=rfc, step=1, cv=5, scoring='f1')
# rfecv_rfc = rfecv_rfc.fit(X_train, y_train)
# print('Optimal number of features :', rfecv_rfc.n_features_)
# print('Best features of logreg :', X_train.columns[rfecv_rfc.support_])
# rfecv_rfc_list = list(X_train.columns[rfecv_rfc.support_])
```

    Optimal number of features : 18
    Best features of logreg : Index(['V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14',
           'V16', 'V17', 'V18', 'V21', 'V22', 'V26', 'Amount'],
          dtype='object')


### SelectFromModel


```python
from sklearn.feature_selection import SelectFromModel
```


```python
select_lr = SelectFromModel(lr)
select_lr = select_lr.fit(X_train, y_train)
```


```python
# save selected features in a list
select_lr_list = list(X_train.columns[select_lr.get_support()])
```


```python
select_rfc = SelectFromModel(rfc)
select_rfc = select_rfc.fit(X_train, y_train)
```


```python
# save selected features in a list
select_rfc_list = list(X_train.columns[select_rfc.get_support()])
```


```python
print('kbest:', kbest_list)
print('rfecv_lr:', rfecv_lr_list)
print('rfecv_rfc:', rfecv_rfc_list)
print('select_lr:', select_lr_list)
print('select_rfc:', select_rfc_list)
```

    kbest: ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 'V11', 'V4', 'V18', 'V1', 'V9', 'V5', 'V2']
    rfecv_lr: ['V1', 'V2', 'V4', 'V6', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V16', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V26', 'V27', 'V28']
    rfecv_rfc: ['V1', 'V2', 'V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V21', 'V22', 'V26', 'Amount']
    select_lr: ['V2', 'V3', 'V9', 'V10', 'V13', 'V14', 'V15', 'V16', 'V17', 'V22']
    select_rfc: ['V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']


Here, I want to select features that appeared in 3 or more feature selection techniques.


```python
# combine multiple lists into 1
ensemble_list = kbest_list + rfecv_lr_list + rfecv_rfc_list + select_lr_list + select_rfc_list
```


```python
from collections import Counter
```


```python
# using counter to count number of occurences per feature in the list
counter_list = Counter(ensemble_list).most_common()
```


```python
# a loop to filter out any features that appeared in 3 or more lists
final_list = []

for i in counter_list:
  if i[1] > 2:
    final_list.append(i[0])
```


```python
final_list
```




    ['V14',
     'V10',
     'V16',
     'V9',
     'V17',
     'V12',
     'V11',
     'V18',
     'V2',
     'V3',
     'V4',
     'V1',
     'V22']




```python
X_train_fs = X_train[final_list]
X_test_fs = X_test[final_list]
```

## Quick Model Run

Without doing any parameter tuning, let's do some comparison between training a model without feature selection vs a model with feature selection.

### Logistic Regression


```python
# logreg without feature selection
logreg_base = LogisticRegression().fit(X_train, y_train)

# logreg with feature selection
logreg_fs = LogisticRegression().fit(X_train_fs, y_train)
```


```python
logreg_base_preds = logreg_base.predict(X_test)
logreg_fs_preds = logreg_fs.predict(X_test_fs)
```


```python
class_names = unique_labels(y_test, logreg_fs_preds)


fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
plot_confusion_matrix(y_test, logreg_base_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n Before Feature Selection')

plt.subplot(1, 2, 2)
plot_confusion_matrix(y_test, logreg_fs_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After Feature Selection')
```


![png](output_52_0.png)



```python
print('before feature selection')
generate_model_report(y_test, logreg_base_preds)
print()
print('after feature selection')
generate_model_report(y_test, logreg_fs_preds)

fig = plt.figure(figsize = (15,15))
plt.subplot(2, 2, 1)
generate_auc_roc_curve(logreg_base, X_test, y_test, title = 'ROC Curve \n Before Feature Selection')

plt.subplot(2, 2, 2)
generate_auc_roc_curve(logreg_fs, X_test_fs, y_test, title = 'ROC Curve \n After Feature Selection')

plt.subplot(2, 2, 3)
generate_pr_curve(logreg_base, X_test, y_test, title = 'PR Curve \n Before Feature Selection')

plt.subplot(2, 2, 4)
generate_pr_curve(logreg_fs, X_test_fs, y_test, title = 'PR Curve \n After Feature Selection')

```

    before feature selection
    Accuracy =  0.9990519995786665
    Precision =  0.6794871794871795
    Recall =  0.6463414634146342
    F1 Score =  0.6625000000000001

    after feature selection
    Accuracy =  0.9993679997191109
    Precision =  0.8484848484848485
    Recall =  0.6829268292682927
    F1 Score =  0.7567567567567567



![png](output_53_1.png)


We can see that doing feature selection clearly improves Logistic Regression model for this dataset, a f1-score increment from `0.6625` to `0.7568`.

### Random Forest


```python
# logreg without feature selection
rfc_base = RandomForestClassifier().fit(X_train, y_train)

# logreg with feature selection
rfc_fs = RandomForestClassifier().fit(X_train_fs, y_train)
```


```python
rfc_base_preds = rfc_base.predict(X_test)
rfc_fs_preds = rfc_fs.predict(X_test_fs)
```


```python
class_names = unique_labels(y_test, rfc_fs_preds)


fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
plot_confusion_matrix(y_test, rfc_base_preds, classes=class_names,
                      title='Vanilla Random Forest Classifier \n Before Feature Selection')

plt.subplot(1, 2, 2)
plot_confusion_matrix(y_test, rfc_fs_preds, classes=class_names,
                      title='Vanilla Random Forest Classifier \n After Feature Selection')
```


![png](output_58_0.png)



```python
print('before feature selection')
generate_model_report(y_test, rfc_base_preds)
print()
print('after feature selection')
generate_model_report(y_test, rfc_fs_preds)

fig = plt.figure(figsize = (15,15))
plt.subplot(2, 2, 1)
generate_auc_roc_curve(rfc_base, X_test, y_test, title = 'ROC Curve \n Before Feature Selection')

plt.subplot(2, 2, 2)
generate_auc_roc_curve(rfc_fs, X_test_fs, y_test, title = 'ROC Curve \n After Feature Selection')

plt.subplot(2, 2, 3)
generate_pr_curve(rfc_base, X_test, y_test, title = 'PR Curve \n Before Feature Selection')

plt.subplot(2, 2, 4)
generate_pr_curve(logreg_fs, X_test_fs, y_test, title = 'PR Curve \n After Feature Selection')
```

    before feature selection
    Accuracy =  0.9997542221129876
    Precision =  0.9722222222222222
    Recall =  0.8536585365853658
    F1 Score =  0.9090909090909091

    after feature selection
    Accuracy =  0.9995962220427653
    Precision =  0.9041095890410958
    Recall =  0.8048780487804879
    F1 Score =  0.8516129032258065



![png](output_59_1.png)


It appears that Random Forest Classifier perform really well on this dataset even without feature selection, with feature selection, the f1-score is improved slightly from `0.8816` to `0.8947`

### Linear Support Vector Machine


```python
from sklearn.svm import LinearSVC
```


```python
# logreg without feature selection
svc_base = LinearSVC().fit(X_train, y_train)

# logreg with feature selection
svc_fs = LinearSVC().fit(X_train_fs, y_train)
```


```python
svc_base_preds = svc_base.predict(X_test)
svc_fs_preds = svc_fs.predict(X_test_fs)
```


```python
class_names = unique_labels(y_test, svc_fs_preds)


fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
plot_confusion_matrix(y_test, svc_base_preds, classes=class_names,
                      title='Vanilla Support Vector Classifier \n Before Feature Selection')

plt.subplot(1, 2, 2)
plot_confusion_matrix(y_test, svc_fs_preds, classes=class_names,
                      title='Vanilla Support Vector Classifier \n After Feature Selection')
```


![png](output_65_0.png)



```python
print('before feature selection')
generate_model_report(y_test, svc_base_preds)
print()
print('after feature selection')
generate_model_report(y_test, svc_fs_preds)

fig = plt.figure(figsize = (15,15))
plt.subplot(2, 2, 1)
y_score = svc_base.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test,  y_score)
auc = roc_auc_score(y_test, y_score)
plt.plot(fpr,tpr,label = 'ROC Curve with AUC =' + str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate / Recall')
plt.title('ROC Curve \n Before Feature Selection')
plt.legend(loc=4)

plt.subplot(2, 2, 2)
y_score = svc_fs.decision_function(X_test_fs)
fpr, tpr, thresholds = roc_curve(y_test,  y_score)
auc = roc_auc_score(y_test, y_score)
plt.plot(fpr,tpr,label = 'ROC Curve with AUC =' + str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate / Recall')
plt.title('ROC Curve \n After Feature Selection')
plt.legend(loc=4)


plt.subplot(2, 2, 3)
generate_pr_curve(rfc_base, X_test, y_test, title = 'PR Curve \n Before Feature Selection')

plt.subplot(2, 2, 4)
generate_pr_curve(logreg_fs, X_test_fs, y_test, title = 'PR Curve \n After Feature Selection')
```

    before feature selection
    Accuracy =  0.9987008883115059
    Precision =  0.7
    Recall =  0.17073170731707318
    F1 Score =  0.2745098039215686

    after feature selection
    Accuracy =  0.9995611109160493
    Precision =  0.8904109589041096
    Recall =  0.7926829268292683
    F1 Score =  0.8387096774193549



![png](output_66_1.png)


Using selected features, `LinearSVM` is able to increase the f1-score from `0.2292` to `0.8312`.

## Class Weight Parameter Tuning

### Logistic Regression


```python
weights = np.linspace(0.05, 0.20, 20)
lr_gs = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=5
)

lr_grid_result = lr_gs.fit(X_train_fs, y_train)
print("Best parameters : %s" % lr_grid_result.best_params_)
```

    Best parameters : {'class_weight': {0: 0.16842105263157897, 1: 0.831578947368421}}



```python
lr_score_weights = pd.DataFrame({'score': lr_grid_result.cv_results_['mean_test_score'],
                       'weight_0': weights })
lr_score_weights.sort_values('score', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>weight_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>0.793203</td>
      <td>0.168421</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.791351</td>
      <td>0.136842</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.791154</td>
      <td>0.176316</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.790931</td>
      <td>0.121053</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.790804</td>
      <td>0.144737</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.790804</td>
      <td>0.152632</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.790431</td>
      <td>0.128947</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.789308</td>
      <td>0.160526</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.789021</td>
      <td>0.113158</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.788114</td>
      <td>0.184211</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.787363</td>
      <td>0.097368</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.786596</td>
      <td>0.192105</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.786037</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.785797</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.784526</td>
      <td>0.081579</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.781158</td>
      <td>0.089474</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.772957</td>
      <td>0.073684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.767564</td>
      <td>0.065789</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.760373</td>
      <td>0.057895</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.751365</td>
      <td>0.050000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lineplot(x = 'weight_0', y = 'score', data = lr_score_weights)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2f1a1c5128>




![png](output_72_1.png)



```python
## refit model with best params
lr_fs_pt = LogisticRegression(**lr_grid_result.best_params_)
```


```python
lr_fs_pt.fit(X_train_fs, y_train)
```




    LogisticRegression(C=1.0,
                       class_weight={0: 0.16842105263157897, 1: 0.831578947368421},
                       dual=False, fit_intercept=True, intercept_scaling=1,
                       l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None,
                       penalty='l2', random_state=None, solver='warn', tol=0.0001,
                       verbose=0, warm_start=False)




```python
lr_fs_pt_preds = lr_fs_pt.predict(X_test_fs)
```

### Random Forest

For Random Forest, I tried parameter tuning for `class_weight` but it actually made the performance worse, so I've decided to tune other parameter.


```python
weights = np.linspace(0.05, 0.95, 30)
rfc_gs = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
#        'class_weight': [{0: x, 1: 1.0-x} for x in weights],
        'n_estimators': [3, 5, 10, 50, 100],
        'min_samples_split': [2, 3, 5, 10]
    },
    scoring='f1',
    cv=5
)

rfc_grid_result = rfc_gs.fit(X_train_fs, y_train)
```


```python
print("Best parameters : %s" % rfc_grid_result.best_params_)
```

    Best parameters : {'min_samples_split': 2, 'n_estimators': 50}



```python
## refit rfc model with best params
rfc_fs_pt = RandomForestClassifier(**rfc_grid_result.best_params_)
```


```python
rfc_fs_pt.fit(X_train_fs, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=50,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
rfc_fs_pt_preds = rfc_fs_pt.predict(X_test_fs)
```

### Linear Support Vector Machine


```python
weights = np.linspace(0.05, 0.20, 20)
svc_gs = GridSearchCV(
    estimator=LinearSVC(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights],
    },
    scoring='f1',
    cv=5
)

svc_grid_result = svc_gs.fit(X_train_fs, y_train)
```


```python
print("Best parameters : %s" % svc_grid_result.best_params_)
```

    Best parameters : {'class_weight': {0: 0.1605263157894737, 1: 0.8394736842105263}}



```python
svc_score_weights = pd.DataFrame({'score': svc_grid_result.cv_results_['mean_test_score'],
                       'weight_0': weights })
svc_score_weights.sort_values('score', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>weight_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>0.820183</td>
      <td>0.160526</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.819154</td>
      <td>0.152632</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.818136</td>
      <td>0.121053</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.818055</td>
      <td>0.113158</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.817610</td>
      <td>0.168421</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.817144</td>
      <td>0.184211</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.817049</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.816815</td>
      <td>0.089474</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.816652</td>
      <td>0.176316</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.816014</td>
      <td>0.128947</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.815943</td>
      <td>0.097368</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.815153</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.815029</td>
      <td>0.144737</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.815029</td>
      <td>0.136842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.814711</td>
      <td>0.081579</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.812365</td>
      <td>0.192105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.808862</td>
      <td>0.073684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.802016</td>
      <td>0.065789</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.794519</td>
      <td>0.057895</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.788569</td>
      <td>0.050000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lineplot(x = 'weight_0', y = 'score', data = svc_score_weights)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2f1cc6c630>




![png](output_87_1.png)



```python
## refit rfc model with best params
svc_fs_pt = LinearSVC(**svc_grid_result.best_params_)
```


```python
svc_fs_pt.fit(X_train_fs, y_train)
```




    LinearSVC(C=1.0, class_weight={0: 0.1605263157894737, 1: 0.8394736842105263},
              dual=True, fit_intercept=True, intercept_scaling=1,
              loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2',
              random_state=None, tol=0.0001, verbose=0)




```python
svc_fs_pt_preds = svc_fs_pt.predict(X_test_fs)
```

## Evaluation

### Confusion Matrix


```python
class_names = unique_labels(y_test, svc_fs_preds)

fig = plt.figure(figsize = (13,15))
plt.subplot(3, 3, 1)
plot_confusion_matrix(y_test, logreg_base_preds, classes=class_names,
                      title='Logistic Regression')

plt.subplot(3, 3, 2)
plot_confusion_matrix(y_test, logreg_fs_preds, classes=class_names,
                      title='Logistic Regression \n After Feature Selection')

plt.subplot(3, 3, 3)
plot_confusion_matrix(y_test, lr_fs_pt_preds, classes=class_names,
                      title='Logistic Regression \n After Feature Selection \n & Parameter Tuning')

plt.subplot(3, 3, 4)
plot_confusion_matrix(y_test, rfc_base_preds, classes=class_names, cmap = plt.cm.Greens,
                      title='Random Forest Classifier')

plt.subplot(3, 3, 5)
plot_confusion_matrix(y_test, rfc_fs_preds, classes=class_names, cmap = plt.cm.Greens,
                      title='Random Forest Classifier \n After Feature Selection')

plt.subplot(3, 3, 6)
plot_confusion_matrix(y_test, rfc_fs_pt_preds, classes=class_names, cmap = plt.cm.Greens,
                      title='Random Forest Classifier \n After Feature Selection \n & Parameter Tuning')

plt.subplot(3, 3, 7)
plot_confusion_matrix(y_test, svc_base_preds, classes=class_names, cmap = plt.cm.Oranges,
                      title='Support Vector Classifier')

plt.subplot(3, 3, 8)
plot_confusion_matrix(y_test, svc_fs_preds, classes=class_names, cmap = plt.cm.Oranges,
                      title='Support Vector Classifier \n After Feature Selection')

plt.subplot(3, 3, 9)
plot_confusion_matrix(y_test, svc_fs_pt_preds, classes=class_names, cmap = plt.cm.Oranges,
                      title='Support Vector Classifier \n After Feature Selection \n & Parameter Tuning')
```


![png](output_93_0.png)


### F1-score


```python
print('Random Forest After Feature Selection & Parameter Tuning')
generate_model_report(y_test, rfc_fs_pt_preds)

fig = plt.figure(figsize = (15,5))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(rfc_base, X_test, y_test, title = 'ROC Curve')

plt.subplot(1, 2, 2)
generate_pr_curve(rfc_fs_pt, X_test_fs, y_test, title = 'PR Curve')
```

    Random Forest After Feature Selection & Parameter Tuning
    Accuracy =  0.9997191109862715
    Precision =  0.9342105263157895
    Recall =  0.8658536585365854
    F1 Score =  0.8987341772151899



![png](output_95_1.png)


Since we do not know the cost associated with type I error and type II error, we won't be able to do cost sensitive learning.

Consequently, we will select the model that gives the highest f1-score `0.8987` which is random forest after feature selection and parameter tuning.
