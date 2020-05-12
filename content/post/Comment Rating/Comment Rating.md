---
date: 2020-5-11
title: Comment Rating System
---


Comment Rating System


```python
import numpy as np
import pandas as pd
import math
import re
import random
from csv import reader
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
```

Step1: Read data
Step2: Preprocessing,Remove the punctuation and stopword and no comment data


```python
data_drict = 'boardgamegeek-reviews/bgg-13m-reviews.csv'
stopwords = (nltk.corpus.stopwords.words('english'))

with open(data_drict,'r',encoding='utf-8') as f:
    row_data = reader(f)
    review = []
    rate = []
    for row in row_data:
        if  row[0] != '' and row[3] !='':
            rate.append(round(float(row[2])))
            content = row[3].lower()
            content = content.replace("\r", "").strip()  
            content = content.replace("\n", "").strip()
            content = re.sub("[%s]+"%('.,|?|!|:|;\"\-|#|$|%|&|\|(|)|*|+|-|/|<|=|>|@|^|`|{|}|~\[\]'), "", content)
            sentence = content.split(' ')
            for i in stopwords:
                while i in sentence:
                    sentence.remove(i)
            content = ' '.join(sentence)
            review.append(content)
```


```python
print('Vaild data size:')
print(len(review))
```

    Vaild data size:
    2638172
    

Step4: Split train and testing data set


```python
x_train,x_test, y_train, y_test = train_test_split(review,rate,test_size=0.3, random_state=0)
```


```python
print('Trainset_size:')
print(len(x_train))
print('Testingset_size:')
print(len(x_test))
```

    Trainset_size:
    1846720
    Testingset_size:
    791452
    


```python
x_0 = x_train[:100000]
y_0 = y_train[:100000]
x_1 = x_train[100000:200000]
y_1 = y_train[100000:200000]
x_2 = x_train[200000:300000]
y_2 = y_train[200000:300000]
x_test = x_test[:100000]
y_test = y_test[:100000]

test_y = np.asarray(y_test)
```

Step5: Train 3 different classifier


```python
vectorizer0 = feature_extraction.text.CountVectorizer()
train0_x = vectorizer0.fit_transform(x_0)
train0_y = np.asarray(y_0)

test_x = vectorizer0.transform(x_test)

NB0 = MultinomialNB()
NB0.fit(train0_x,train0_y)

pred0 = NB0.predict(test_x)
acc_0 = accuracy_score(test_y,pred0)
```


```python
print('The first NB classifier precision is '+ str(acc_0))
```

    The first NB classifier precision is 0.3013
    


```python
vectorizer1 = feature_extraction.text.CountVectorizer()
train1_x = vectorizer1.fit_transform(x_1)
train1_y = np.asarray(y_1)

test_x = vectorizer1.transform(x_test)

NB1 = MultinomialNB()
NB1.fit(train1_x,train1_y)

pred1 = NB1.predict(test_x)
acc_1 = accuracy_score(test_y,pred1)
```


```python
print('The second NB classifier precision is '+ str(acc_1))
```

    The second NB classifier precision is 0.29945
    


```python
vectorizer2 = feature_extraction.text.CountVectorizer()
train2_x = vectorizer2.fit_transform(x_2)
train2_y = np.asarray(y_2)

test_x = vectorizer2.transform(x_test)

NB2 = MultinomialNB()
NB2.fit(train2_x,train2_y)

pred2 = NB2.predict(test_x)
acc_1 = accuracy_score(test_y,pred2)

```


```python
print('The third NB classifier precision is '+ str(acc_2))
```

    The third NB classifier precision is 0.30012
    


```python
result = []
for i in range(len(test_y)):
    pred = []
    pred.append(pred0[i])
    pred.append(pred1[i])
    pred.append(pred2[i])
    tmp = dict((a, pred.count(a)) for a in pred)
    top = sorted(tmp.items() , key=lambda tmp:tmp[1] ,reverse= True)
    result.append(top[0][0])
```


```python
acc = accuracy_score(test_y,result)
```


```python
print('The combining NB classifier precision is '+str(acc))
```

    The combining NB classifier precision is 0.30734
    


```python
result1 = []
for i in range(len(test_y)):
    Min = test_y[i] - 1.1
    Max = test_y[i] + 1.1
    if result[i]>Min and result[i]<Max:
        pd = 1
    else:
        pd = 0
    result1.append(pd)
```


```python
final_acc1 = result1.count(1) / len(test_y)
print('If we keep the rating error of plus or minus one:')
print('Final precision is '+str(final_acc1))
```

    If we keep the rating error of plus or minus one:
    Final precision is 0.67312
    


```python
result2 = []
for i in range(len(test_y)):
    Min = test_y[i] - 2.1
    Max = test_y[i] + 2.1
    if result[i]>Min and result[i]<Max:
        pd = 1
    else:
        pd = 0
    result2.append(pd)

final_acc2 = result2.count(1) / len(test_y)
print('If we keep the rating error of plus or minus two:')
print('Final precision is '+str(final_acc2))
```

    If we keep the rating error of plus or minus two:
    Final precision is 0.88714
    
