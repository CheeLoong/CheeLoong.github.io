---
title: "Parameter Tuning & Feature Selection on Credit Card Fraud dataset"
date: 2019-08-28
permalink: /ccfraud-ptfs
tags: [parameter tuning, feature selection, logistic regression, credit card fraud]
excerpt: "Exploring techniques to deal with imbalanced dataset and doing feature selection"
mathjax: "true"
---

Hello world, in this blogpost we will exploring the same old credit card dataset that I have previously built a neural network around.

This time however, we will talk about these things:
- `class_weight` parameter tuning (Deal with class-imbalanced)
- SMOTE oversampling (Deal with class-imbalanced)
- Univariate feature selection (Feature Selection)
- Recursive feature Elimination (Feature Selection)


Note that this blogpost is focused on the above topics and so we will only be using Logistic Regression as the model for demonstration, in the future post, I will compare this model with other models and find out the best performing model.

## Libraries and Modules

Here's I am just going to import some standard `sklearn` modules that I will be using, and also functions to plot confusion matrix, and a function to print out the evaluation metrics for this dataset.


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

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
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
    y_score = clf.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test,  y_score)
    aupc = average_precision_score(y_test, y_score)
    plt.plot(recall,precision,label = 'PR Curve with AUC =' + str(aupc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc=4)
```

## Look at data

Importing the data now and let's look at the distribution of the class.


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




```python
df.__ge
```


```python
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
# Graph
my_pal = {0: 'deepskyblue', 1: 'deeppink'}

plt.figure(figsize = (8, 5))
ax = sns.countplot(x = df['Class'], palette = my_pal, edgecolor="0.1")
plt.title('Credit Card Fraud \n Class Distribution')
plt.show()

# Count and %
Count_Normal_transacation = len(df[df['Class']==0])
Count_Fraud_transacation = len(df[df['Class']==1])

Percentage_of_Normal_transacation = Count_Normal_transacation/(Count_Normal_transacation+Count_Fraud_transacation)
print('Percentage of normal transacation       :', "{0:.2f}%".format(Percentage_of_Normal_transacation*100))
print('Number of normal transaction            :', Count_Normal_transacation)

Percentage_of_Fraud_transacation= Count_Fraud_transacation/(Count_Normal_transacation+Count_Fraud_transacation)
print('Percentage of fraud transacation        :', "{0:.2f}%".format(Percentage_of_Fraud_transacation*100))
print('Number of fraud transaction             :', Count_Fraud_transacation)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_16_0.png" alt="">


    Percentage of normal transacation       : 99.83%
    Number of normal transaction            : 284315
    Percentage of fraud transacation        : 0.17%
    Number of fraud transaction             : 492


## Train Test Split

I have decided to allocate 20% of the data as test set and the rest as train set.


```python
from sklearn.model_selection import train_test_split
```


```python
X = df.drop('Class', axis=1)
y = df['Class']
```


```python
# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
sum(y_test == 1) / len(y_test)
```




    0.0017204452090867595




```python
len(y_test == 1)
```




    56962




```python
sum(y_test == 1)
```




    98



## Vanilla Logistic Regression

Let's try to fit a vanilla logistic regression model and see how well it performs.


```python
from sklearn.linear_model import LogisticRegression
```


```python
logreg = LogisticRegression()
```


```python
logreg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
logreg_preds = logreg.predict(X_test)
```


```python
# logreg_cm = confusion_matrix(y_test, logreg_preds)
# logreg_cm
```


```python
class_names = unique_labels(y_test, logreg_preds)

plot_confusion_matrix(y_test, logreg_preds, classes=class_names,
                      title='Vanilla Logistic Regression')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba9ce40a90>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_32_1.png" alt="">



```python
generate_model_report(y_test, logreg_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(logreg, X_test, y_test, title = 'ROC Curve \n class_weights = None')

plt.subplot(1, 2, 2)
generate_pr_curve(logreg, X_test, y_test, title = 'PR Curve \n class_weights = None')
```

    Accuracy =  0.9989993328885924
    Precision =  0.8253968253968254
    Recall =  0.5306122448979592
    F1 Score =  0.6459627329192548



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_33_1.png" alt="">


Without doing any pre-processing, our baseline vanilla logistic regression perform pretty badly. 0.5306 recall means the model is only able to capture 53.06% of the actual fraud transactions, and out of all the fraud predictions, 82.54% were spot on, and the f1 is 0.646.

**Two methods to deal with class-imbalanced dataset**

There are two methods that I am aware of that can potentially increases the f1 score, the first is by tuning the `class_weight` parameter that is available in most of the classifiers from `sk.learn`, which essentially assign different weights on different classes so there will be a heavy penalization on misclassification on the minority class.

Another method is by creating a pipeline which take a resampling technique and a classifier, by upsampling the minority class, we are essentially making the class-imbalance issue less serious.

Both methods are basically trying to deal with the imbalanced class issue, so let us explore both of these methods and see which one works better on this dataset.

## Class Weight

### Brief explanation of `class_weight`

Here, I am going to discuss about this parameter called `class_weight`, and what it means if we set the argument `balanced` to it.

The `class_weight` parameter is a parameter that is used to allocate weights to the classes in the dataset, so the idea is that if we have an imbalanced dataset, we can assign a higher weight to the minority class (e.g. in our case, the fraud class), by doing this, the optimization process will then pay heavy attention to not misclassify the minority class due to the high weightage associated with it.

The mathematical expression of the `class_weight` parameter look like this:

$$ w_j = \frac{n}{k * n_j} $$

where $w_j$ is ther weight of class j, and $n_j$ is the number of observations in class j, $n$ is the total number of observations, and k is the total number of classes.

This formula essentially return higher weight for minority class and lower weight for majority class, let's try to compute it.


```python
unique_classes = list(df['Class'].unique())
unique_classes
```




    [0, 1]




```python
cw_dict = {}
for classes in unique_classes:
    cw_dict[classes] = df.shape[0]/((df.loc[df['Class'] == classes].shape[0])
                                     *len(unique_classes))
```


```python
cw_dict
```




    {0: 0.5008652375006595, 1: 289.4380081300813}



As you can see, fraud class is our minority class and it is assigned a very high weight.

### Setting `class_weight = 'balanced'`


```python
logreg = LogisticRegression(class_weight = 'balanced')
```


```python
logreg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# This is the weight assigned to class 0 and class 1
class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
```




    array([  0.50086612, 289.14340102])




```python
logreg_preds = logreg.predict(X_test)
```


```python
class_names = unique_labels(y_test, logreg_preds)

plot_confusion_matrix(y_test, logreg_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n class_weight = "balanced"')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba9a43ccf8>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_47_1.png" alt="">



```python
generate_model_report(y_test, logreg_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(logreg, X_test, y_test, title = 'ROC Curve \n class_weights = None')

plt.subplot(1, 2, 2)
generate_pr_curve(logreg, X_test, y_test, title = 'PR Curve \n class_weights = None')
```

    Accuracy =  0.9764404339735262
    Precision =  0.06320224719101124
    Recall =  0.9183673469387755
    F1 Score =  0.11826544021024968



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_48_1.png" alt="">


As we can see, the precision of the model got a lot worse, but the recall got a lot better, this is because we assigned a higher weight on the minority class, thus, a misclassfication on the minority class will be penalized heavily. Consequently, the model will lean towards classifying observations as 'fraud', and that is why the model now has a lot more false positives than before.

But here's the thing, overall AUROC and AUPRC has increased? So is the model actually better or worse than before? [This post](https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc) basically explain AUROC and F1 score and which should be emphasized in an imbalanced dataset scenario. F1 score should be prioritized over AUROC or AUPRC, so let's do that.

### Fine tuning `class_weight` parameter with `GridSearchCV`

I am going to do a gridsearch on different `class_weight` and try to find one that maximizes the F1-score.

Since we are doing parameter tuning in this step, its really up to us to decide which parameter we want to tune, so it's completely subjective, but I am just going to tune the `class_weights` and I will be testing 20 different combination of weights and find the one that maximizes the f1-score.

Note that I am limiting the weight of class 0 to be between 0.05 to 0.20, because this is the majority class, and it's really making the computational expense higher when we put a higher weight on it, it makes intuitive sense to keep it low, and i've decided to keep it at most at 0.20 for the cross validation.


```python
weights = np.linspace(0.05, 0.20, 20)
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=5
)

grid_result = gsc.fit(X_train, y_train)
# print("Best parameters : %s" % grid_result.best_params_)
```


```python
score_weights = pd.DataFrame({'score': grid_result.cv_results_['mean_test_score'],
                       'weight_0': weights })
score_weights.sort_values('score', ascending = False)
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
      <th>17</th>
      <td>0.783877</td>
      <td>0.184211</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.778748</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.776592</td>
      <td>0.144737</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.767201</td>
      <td>0.160526</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.762306</td>
      <td>0.121053</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.757674</td>
      <td>0.113158</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.757361</td>
      <td>0.073684</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.757097</td>
      <td>0.192105</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.756442</td>
      <td>0.081579</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.756094</td>
      <td>0.097368</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.755414</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.753629</td>
      <td>0.168421</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.751738</td>
      <td>0.176316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.738889</td>
      <td>0.057895</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.738547</td>
      <td>0.065789</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.732796</td>
      <td>0.152632</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.730282</td>
      <td>0.136842</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.728396</td>
      <td>0.128947</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.726481</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.722379</td>
      <td>0.089474</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lineplot(x = 'weight_0', y = 'score', data = score_weights)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0cb6682fd0>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_54_1.png" alt="">


From the GridSearch with 5 Cross Validation above, we found that using a class weight of `0.2394` for class 0 yields the best f1 score.

Let's fit a new model with the proposed class weights.


```python
logreg = LogisticRegression(**grid_result.best_params_)
```


```python
logreg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0,
                       class_weight={0: 0.1842105263157895, 1: 0.8157894736842105},
                       dual=False, fit_intercept=True, intercept_scaling=1,
                       l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None,
                       penalty='l2', random_state=None, solver='warn', tol=0.0001,
                       verbose=0, warm_start=False)




```python
logreg_preds = logreg.predict(X_test)
```


```python
class_names = unique_labels(y_test, logreg_preds)

plot_confusion_matrix(y_test, logreg_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After Tuning for "class_weight"')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0cb65a1470>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_59_1.png" alt="">



```python
generate_model_report(y_test, logreg_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(logreg, X_test, y_test, title = 'ROC Curve \n After Tuning for "class_weight"')

plt.subplot(1, 2, 2)
generate_pr_curve(logreg, X_test, y_test, title = 'PR Curve \n After Tuning for "class_weight"')
```

    Accuracy =  0.9991748885221726
    Precision =  0.7741935483870968
    Recall =  0.7346938775510204
    F1 Score =  0.7539267015706805



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_60_1.png" alt="">


Alright so by tuning the `class_weight` parameter, we manage to improve the F1-score of the model from `0.646` to `0.754`, that is pretty good, can we beat that with resampling techniques? Let's try!

## Resampling Techniques

In this section, people usually try random undersampling, random oversampling, and SMOTE oversampling.

Due to time constraint and also some reading on each of the techniques, I have decided to try SMOTE oversampling because it's more emperically proven than others.  

### Sythentic Minority Oversampling Technique (SMOTE)

Remember to upsample only on the train set, we do not want synthetic data on our test set when we evaluate the test accuracy, so all the resampling is strictly on the train set only!


```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
```


```python
unique, count = np.unique(y_train, return_counts=True)
y_train_dict_value_count = { k:v for (k,v) in zip(unique, count)}
y_train_dict_value_count
```




    {0: 227451, 1: 394}




```python
sm = SMOTE(random_state=69, ratio = 1)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
```

the `ratio` parameter takes either string or float as an argument, if in float, it must be in range between 0 and 1, which determines how much sythentic data are we replicating on the minority classes, let's try the max value of 1, such that class is equally distributed.


```python
unique, count = np.unique(y_train_res, return_counts=True)
y_train_smote_value_count = { k:v for (k,v) in zip(unique, count)}
y_train_smote_value_count
```




    {0: 227451, 1: 227451}




```python
sm_logreg = LogisticRegression()
```


```python
sm_logreg.fit(X_train_res, y_train_res)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
sm_logreg_preds = sm_logreg.predict(X_test)
```


```python
class_names = unique_labels(y_test, sm_logreg_preds)

plot_confusion_matrix(y_test, sm_logreg_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After Tuning for "class_weight"')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc68ccae3c8>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_73_1.png" alt="">



```python
generate_model_report(y_test, sm_logreg_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(sm_logreg, X_test, y_test, title = 'ROC Curve \n After SMOTE Up-sampling')

plt.subplot(1, 2, 2)
generate_pr_curve(sm_logreg, X_test, y_test, title = 'PR Curve \n After SMOTE Up-sampling')
```

    Accuracy =  0.9833924370633054
    Precision =  0.0859375
    Recall =  0.8979591836734694
    F1 Score =  0.1568627450980392



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_74_1.png" alt="">


The pattern sounds a bit familiar, precision and recall trade off is bad causing f1-score to be very low, let's tune parameter `ratio` to test on different amount of synthetic points to generate.  


```python
pipe = make_pipeline(
    SMOTE(),
    LogisticRegression()
)

weights = np.linspace(0.002, 0.8, 20)

gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        'smote__ratio': weights
    },
    scoring='f1',
    cv=5
)
grid_result = gsc.fit(X_train, y_train)

print("Best parameters : %s" % grid_result.best_params_)
weight_f1_score_df = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                                   'weight': weights })
weight_f1_score_df.plot(x='weight')
```

    Best parameters : {'smote__ratio': 0.002}





    <matplotlib.axes._subplots.AxesSubplot at 0x7fba992da9b0>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_76_2.png" alt="">


The best smote ratio is the minimum of the range we specified, and looking at the graph of f1 score vs smote ratio, it looks like the more SMOTE upsampling we do, the worse f1-score we are going to get.

Let's repeat the procedure with a much smaller range, and see if we can find anything


```python
pipe = make_pipeline(
    SMOTE(),
    LogisticRegression()
)

weights = np.linspace(0.002, 0.005, 20)

gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        'smote__ratio': weights
    },
    scoring='f1',
    cv=5
)
grid_result = gsc.fit(X_train, y_train)

print("Best parameters : %s" % grid_result.best_params_)
weight_f1_score_df = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                                   'weight': weights })
weight_f1_score_df.plot(x='weight')
```

    Best parameters : {'smote__ratio': 0.0040526315789473685}





    <matplotlib.axes._subplots.AxesSubplot at 0x7fba98ffee10>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_78_2.png" alt="">



```python
pipe = make_pipeline(
    SMOTE(ratio=0.0040526),
    LogisticRegression()
)

pipe.fit(X_train, y_train)

pipe_preds = pipe.predict(X_test)
```


```python
class_names = unique_labels(y_test, pipe_preds)

plot_confusion_matrix(y_test, pipe_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After Tuning for SMOTE "ratio"')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba99039668>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_80_1.png" alt="">



```python
generate_model_report(y_test, pipe_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(pipe, X_test, y_test, title = 'ROC Curve \n After tuning for SMOTE ratio')

plt.subplot(1, 2, 2)
generate_pr_curve(pipe, X_test, y_test, title = 'PR Curve \n After tuning for SMOTE ratio')
```

    Accuracy =  0.9991573329588147
    Precision =  0.8048780487804879
    Recall =  0.673469387755102
    F1 Score =  0.7333333333333334



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_81_1.png" alt="">


`0.7333` f1-score, it is pretty decent, but it is still worse than the `class_weight` approach which gives us `0.7539`. This tells us that the `class_weight` approach has a better trade-off between recall and precision than SMOTE oversampling technique.

Depending on the cost of False Negatives and False Positives, we might choose to go with the SMOTE oversampling technique just because it has a higher precision. Usually, we would do cost-sensitive learning to find out the models that minimizes the cost.

## Feature Selection

Let's talk about feature selection real quick, this dataset has undergone PCA transformation, so it makes sense to say that the unknown principal components are all important and consequently we don't need to do feature selection.

However, I will still do it, just for experimentation purpose, who knows we might actually improve the model.


### Univariate feature selection (Score-based)


```python
from sklearn.feature_selection import SelectKBest, f_classif
```


```python
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
X = df.drop('Class', axis=1)
y = df['Class']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# categorical features in our dataset
categorical_feature_columns = list(set(X_train.columns) - set(X_train._get_numeric_data().columns))
categorical_feature_columns
```




    []




```python
# numerical features in our dataset
numerical_feature_columns = list(X_train._get_numeric_data().columns)
numerical_feature_columns
```




    ['Time',
     'V1',
     'V2',
     'V3',
     'V4',
     'V5',
     'V6',
     'V7',
     'V8',
     'V9',
     'V10',
     'V11',
     'V12',
     'V13',
     'V14',
     'V15',
     'V16',
     'V17',
     'V18',
     'V19',
     'V20',
     'V21',
     'V22',
     'V23',
     'V24',
     'V25',
     'V26',
     'V27',
     'V28',
     'Amount']



Since we have negative value features, we will be using `f_classif` as the scoring function


```python
#apply SelectKBest class to extract top 'k' best features
univar = SelectKBest(score_func=f_classif, k=10)
univar = univar.fit(X_train,y_train)
```

Notice here I put `k=10`, but I don't really know how many features I am trying to select, I usually would first look at the scores associated with each feature to decide on a value on `k`.


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
      <td>27240.806212</td>
    </tr>
    <tr>
      <th>14</th>
      <td>V14</td>
      <td>23547.660704</td>
    </tr>
    <tr>
      <th>12</th>
      <td>V12</td>
      <td>16985.532943</td>
    </tr>
    <tr>
      <th>10</th>
      <td>V10</td>
      <td>11096.188557</td>
    </tr>
    <tr>
      <th>16</th>
      <td>V16</td>
      <td>9239.509953</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V3</td>
      <td>8425.516128</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V7</td>
      <td>7782.311282</td>
    </tr>
    <tr>
      <th>11</th>
      <td>V11</td>
      <td>5680.321617</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V4</td>
      <td>4173.023409</td>
    </tr>
    <tr>
      <th>18</th>
      <td>V18</td>
      <td>2865.506910</td>
    </tr>
    <tr>
      <th>9</th>
      <td>V9</td>
      <td>2172.540570</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V1</td>
      <td>2088.974268</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V5</td>
      <td>1857.295548</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>1822.617413</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V6</td>
      <td>456.079147</td>
    </tr>
    <tr>
      <th>21</th>
      <td>V21</td>
      <td>400.809787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>V19</td>
      <td>288.201333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>V8</td>
      <td>139.228852</td>
    </tr>
    <tr>
      <th>27</th>
      <td>V27</td>
      <td>88.044329</td>
    </tr>
    <tr>
      <th>20</th>
      <td>V20</td>
      <td>72.748650</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Time</td>
      <td>40.596070</td>
    </tr>
    <tr>
      <th>28</th>
      <td>V28</td>
      <td>26.288500</td>
    </tr>
    <tr>
      <th>24</th>
      <td>V24</td>
      <td>12.149344</td>
    </tr>
    <tr>
      <th>23</th>
      <td>V23</td>
      <td>8.171840</td>
    </tr>
    <tr>
      <th>26</th>
      <td>V26</td>
      <td>6.733176</td>
    </tr>
    <tr>
      <th>15</th>
      <td>V15</td>
      <td>4.325969</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Amount</td>
      <td>3.302948</td>
    </tr>
    <tr>
      <th>25</th>
      <td>V25</td>
      <td>1.019057</td>
    </tr>
    <tr>
      <th>13</th>
      <td>V13</td>
      <td>0.943039</td>
    </tr>
    <tr>
      <th>22</th>
      <td>V22</td>
      <td>0.154625</td>
    </tr>
  </tbody>
</table>
</div>



There's really no specific rule that can decide the threshold on the scores, but looking at the table, my first instinct is to remove features with less than 100 score.


```python
selected_features = feature_scores_df[feature_scores_df['Scores'] > 100].sort_values(by='Scores', ascending=False)
```


```python
selected_features.shape[0]
```




    18



From here, we see that 18 features satisfy the threshold we have chosen, so let's input `k=18`


```python
univar = SelectKBest(score_func=f_classif, k=18)
univar = univar.fit(X_train,y_train)
```


```python
X_train_univar = univar.transform(X_train)
X_test_univar = univar.transform(X_test)
```


```python
logreg = LogisticRegression()
logreg.fit(X_train_univar, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
logreg_ufc_preds = logreg.predict(X_test_univar)
```


```python
class_names = unique_labels(y_test, logreg_ufc_preds)

plot_confusion_matrix(y_test, logreg_ufc_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After feature selection with KBest')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba98c1a0f0>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_105_1.png" alt="">



```python
generate_model_report(y_test, logreg_ufc_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(logreg, X_test_univar, y_test, title = 'ROC Curve \n after KBest feature selection')

plt.subplot(1, 2, 2)
generate_pr_curve(logreg, X_test_univar, y_test, title = 'PR Curve \n after KBest feature selection')
```

    Accuracy =  0.9991046662687406
    Precision =  0.8615384615384616
    Recall =  0.5714285714285714
    F1 Score =  0.6871165644171779



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_106_1.png" alt="">


The f1-score is `0.6871` which is higher than our baseline Logistic Regression model without feature selection, not only that, the AUC under ROC Curve and PR Curve also improved after feature selection. It seems like we are headed to the right direction, but let's first talk about another feature selection technique called Recursive Feature Elimination.

### Recursive Feature Elimination


```python
from sklearn.feature_selection import RFECV
```


```python
logreg = LogisticRegression()
```


```python
rfecv = RFECV(estimator=logreg, step=1, cv=5, scoring='f1')
rfecv = rfecv.fit(X_train, y_train)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])
```

    Optimal number of features : 19
    Best features : Index(['V1', 'V2', 'V4', 'V6', 'V8', 'V9', 'V10', 'V13', 'V14', 'V15', 'V16',
           'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V27', 'V28'],
          dtype='object')



```python
rfecv.grid_scores_
```




    array([0.59527604, 0.61213661, 0.64679311, 0.6621501 , 0.69700719,
           0.69487473, 0.69744957, 0.70942647, 0.71731364, 0.72316436,
           0.72589008, 0.7198366 , 0.72184726, 0.72334071, 0.72378882,
           0.72774193, 0.72641568, 0.72460062, 0.72868008, 0.72868008,
           0.72392164, 0.72392164, 0.72367929, 0.72367929, 0.72163696,
           0.7235106 , 0.72030385, 0.72030385, 0.72351883, 0.68502983])




```python
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_113_0.png" alt="">


By using Recursive feature elimination with cross validation, the model has selected 19 features that return the highest f1-score.


```python
X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)
```


```python
logreg = LogisticRegression()
logreg.fit(X_train_rfecv, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
logreg_rfecv_preds = logreg.predict(X_test_rfecv)
```


```python
class_names = unique_labels(y_test, logreg_rfecv_preds)

plot_confusion_matrix(y_test, logreg_rfecv_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After feature selection with KBest')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba98db8cc0>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_118_1.png" alt="">



```python
generate_model_report(y_test, logreg_rfecv_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(logreg, X_test_rfecv, y_test, title = 'ROC Curve \n after Recursive feature elimination')

plt.subplot(1, 2, 2)
generate_pr_curve(logreg, X_test_rfecv, y_test, title = 'PR Curve \n after Recursive feature elimination')
```

    Accuracy =  0.9991222218320986
    Precision =  0.8636363636363636
    Recall =  0.5816326530612245
    F1 Score =  0.6951219512195121



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_119_1.png" alt="">


`0.6951` f1-score, this beats the baseline model f1-score and is also slightly above the model after univariate feature selection.



### Ensemble Feature Selection

Purely for experimentation purpose, I was thinking to use both feature selection techniques instead of choosing just one, so what I have in mind is to choose features that are selected by both feature selection techniques, I will name the list `intersect_list` since it is the intersection between the feature list from KBest and RFE.


```python
kbest_list = list(selected_features['Feature'])
```


```python
rfe_list = list(X_train.columns[rfecv.support_])
```


```python
# these are the variables that are selected by both approach
intersect_list = [value for value in kbest_list if value in rfe_list]
intersect_list
```




    ['V14', 'V10', 'V16', 'V4', 'V9', 'V1', 'V2', 'V6', 'V21', 'V8']




```python
X_train_intersect = X_train[intersect_list]
X_test_intersect = X_test[intersect_list]
```


```python
logreg = LogisticRegression()
logreg.fit(X_train_intersect, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
logreg_intersect_preds = logreg.predict(X_test_intersect)
```


```python
class_names = unique_labels(y_test, logreg_intersect_preds)

plot_confusion_matrix(y_test, logreg_intersect_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After feature selection with KBest & Recursive Feature Elimination')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba99234c50>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_129_1.png" alt="">



```python
generate_model_report(y_test, logreg_intersect_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(logreg, X_test_intersect, y_test, title = 'ROC Curve \n after KBest & Recursive feature elimination')

plt.subplot(1, 2, 2)
generate_pr_curve(logreg, X_test_intersect, y_test, title = 'PR Curve \n after KBest & Recursive feature elimination')
```

    Accuracy =  0.9991222218320986
    Precision =  0.8529411764705882
    Recall =  0.5918367346938775
    F1 Score =  0.6987951807228915



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_130_1.png" alt="">


`0.6988` f1-score, that's actually a slightly better score than using just recursive feature elimination (RFE) for feature selection. To be honest, I never thought it would work this well since we have eliminated quite some number of variables, from 19 features (in the RFE) to just 10.

There are other feature selection techniques that I have not gone through in this blogpost, if you know more feature selection techniques, feel free to do that, and try to do an 'ensemble' feature selection, just to see if it might gives a better result.

## Back to Class Weight

The reason why I did class weight paramater tuning without feature selection before was because I wanted to see how well the model would perform without feature selection, it's purely for comparison purpose.

Now that we have selected the features, let's try to tune for the `class_weight` parameter once again.


```python
weights = np.linspace(0.05, 0.20, 20)
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=5
)

grid_result = gsc.fit(X_train_intersect, y_train)
print("Best parameters : %s" % grid_result.best_params_)
```

    Best parameters : {'class_weight': {0: 0.12105263157894738, 1: 0.8789473684210526}}



```python
score_weights = pd.DataFrame({'score': grid_result.cv_results_['mean_test_score'],
                       'weight_0': weights })
score_weights.sort_values('score', ascending = False)
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
      <th>9</th>
      <td>0.794129</td>
      <td>0.121053</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.793656</td>
      <td>0.113158</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.793588</td>
      <td>0.128947</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.793488</td>
      <td>0.168421</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.792870</td>
      <td>0.184211</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.792618</td>
      <td>0.105263</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.792066</td>
      <td>0.136842</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.791560</td>
      <td>0.144737</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.791504</td>
      <td>0.160526</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.791274</td>
      <td>0.192105</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.791003</td>
      <td>0.152632</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.790817</td>
      <td>0.176316</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.789752</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.785711</td>
      <td>0.097368</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.781224</td>
      <td>0.081579</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.780552</td>
      <td>0.089474</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.776186</td>
      <td>0.073684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.768297</td>
      <td>0.065789</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.758438</td>
      <td>0.057895</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.752171</td>
      <td>0.050000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.lineplot(x = 'weight_0', y = 'score', data = score_weights)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba99042cf8>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_136_1.png" alt="">



```python
logreg = LogisticRegression(**grid_result.best_params_)
```


```python
logreg.fit(X_train_intersect, y_train)
```




    LogisticRegression(C=1.0,
                       class_weight={0: 0.12105263157894738, 1: 0.8789473684210526},
                       dual=False, fit_intercept=True, intercept_scaling=1,
                       l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None,
                       penalty='l2', random_state=None, solver='warn', tol=0.0001,
                       verbose=0, warm_start=False)




```python
logreg_inter_preds = logreg.predict(X_test_intersect)
```


```python
class_names = unique_labels(y_test, logreg_inter_preds)

plot_confusion_matrix(y_test, logreg_inter_preds, classes=class_names,
                      title='Vanilla Logistic Regression \n After feature selection with KBest & RFE \n & Parameter Tuning for "class_weight"')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fba9888f4e0>




<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_140_1.png" alt="">



```python
generate_model_report(y_test, logreg_inter_preds)

fig = plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
generate_auc_roc_curve(logreg, X_test_intersect, y_test, title = 'ROC Curve \n After feature selection with KBest & RFE \n & Parameter Tuning for "class_weight"')

plt.subplot(1, 2, 2)
generate_pr_curve(logreg, X_test_intersect, y_test, title = 'PR Curve \n After feature selection with KBest & RFE \n & Parameter Tuning for "class_weight"')
```

    Accuracy =  0.9993153330290369
    Precision =  0.8172043010752689
    Recall =  0.7755102040816326
    F1 Score =  0.7958115183246073



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/cc-fraud-ptfs/output_141_1.png" alt="">


`0.7958` f1-score, the feature selection actually helped, this is now our new best f1-score as opposed to the previous `0.7539` f1-score without feature selection.

Thank you for reading this blogpost, I will explore more models and feature selection techniques in the next blogpost.

Be awesome till then!
