---
date: 2020-4-20
title: Assignment3
---


Weixiao Sang

Id:1001780927


```python
import numpy as np
import os
import re
import math
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
```

a.Divide the dataset as train, development and test. 

b.Build a vocabulary as list. 


```python
train_pos_drict ='aclImdb/train/pos'
train_neg_drict ='aclImdb/train/neg'
test_pos_drict ='aclImdb/test/pos'
test_neg_drict ='aclImdb/test/neg'

def split(path):
        cate_list = os.listdir(path)
        class_file = []
        for i in cate_list:
            content_path = path + '/' + i
            content = open(content_path,'r',encoding='utf-8').read().lower()
            content = content.replace("\r", "").strip()  
            content = content.replace("\n", "").strip()
            content = re.sub("[%s]+"%('.,|?|!|:|;\"\-|#|$|%|&|\|(|)|*|+|-|/|<|=|>|@|^|`|{|}|~'), "", content)
            sentence = content.split(' ')
            class_file.append(sentence)
        return class_file

#trian_data
train_pos = split(train_pos_drict)[:10000]
train_pos_num = len(train_pos)
train_neg = split(train_neg_drict)[:10000]
train_neg_num =len(train_neg)
print("Train_data_num: "+"pos"+str(train_pos_num)+"  neg"+str(train_neg_num))
#dev_data
dev_pos = split(train_pos_drict)[10000:]
dev_pos_num = len(dev_pos)
dev_neg = split(train_neg_drict)[10000:]
dev_neg_num =len(dev_neg)
print("Dev_data_num: "+"pos"+str(dev_pos_num)+"  neg"+str(dev_neg_num))
#test_data
test_pos = split(test_pos_drict)
test_pos_num = len(test_pos)
test_neg = split(test_neg_drict)
test_neg_num =len(test_neg)
print("Test_data_num: "+"pos"+str(test_pos_num)+"  neg"+str(test_neg_num))
```

    Train_data_num: pos10000  neg10000
    Dev_data_num: pos2500  neg2500
    Test_data_num: pos12500  neg12500
    


```python
stopword = ['did', 'such', 'doing', 'down', 'me','just', 'very', 'against', 't', "you're", 
          'only', "haven't", 'yours', 'you', 'its', 'other', 'we', 'where', 'then', 'they', 'won', "you've",
          'some', 've', 'y', 'each', "you'll", 'them', 'to', 'was', 'once', 'and', 'ain', 'under', 'through',
          'for', "won't", 'mustn', 'a', 'are', 'that', 'at', 'why', 'any', 'nor', 'these', 'yourselves',
          'has', 'here', "needn't", 'm', 'above', 'up', 'more', 'if', 'ma', 'didn', 'whom', 'can', 'have',
          'an', 'should', 'there', 'couldn', 'her', 'how', 'of', 'doesn', "shouldn't", 'further', 
          "wasn't", 'between', 'd', 'wouldn', 'his', 'being', 'do', 'when', 'hasn', "she's", 'by', "should've",
          'into', 'aren', 'weren', 'as', 'needn', 'what', "it's", 'hadn', 'with', 'after', 'he', 'off', 'not',
          'does', 'own', "weren't", "isn't", 'my', 'too', "wouldn't", 'been', 'again', 'same', 'few', "don't",
          'our', 'myself', 'your', 'before', 'about', 'most', 'during', 'll', 'on', 'shouldn', 'is', 'out',
          "shan't", 'below', 'which', 'from', 'she', 'were', 'those', 'over', 'until', 'theirs', 'mightn',
          'yourself', 'i', 'am', 'so', 'himself', 'it', 'had', 'or', 'all', 'while', "aren't", 'ours',
          "that'll", 'but', 'because', 'in', 'now', 'themselves', 'him', "doesn't", 'both', 're', 'wasn',
          's', "hasn't", "didn't", 'their', "mustn't", 'herself', 'the', 'this', 'will', 'isn', "you'd", 
          'haven', 'itself', "couldn't", 'o', 'be', 'don', 'hers', "mightn't", 'having', "hadn't", 'ourselves',
          'who', 'than', 'br','']
```


```python
word_set = set()
def word_num(data):
    dict = {}
    num = 0
    for sentence in data:
        num += len(sentence)
        for word in sentence:
            word_set.add(word)
            if word not in dict:
                dict[word] = 1
            elif word in dict:
                dict[word] += 1
    return dict,num
#build 2 vocabulary : pos_word_vocabulary,neg_word_vocabulary
pos_word,pos_word_num = word_num(train_pos)
neg_word,neg_word_num = word_num(train_neg)

#omit rare word which occurrence is less than five times

def omit(num:int,voc):
    rare_word = []
    rare_words_num = 0
    for word in voc:
        if voc[word] <= num:
            rare_words_num += voc[word]
            rare_word.append(word)
    for i in rare_word:
        voc.pop(i)
    for i in stopword:
        if i in voc:
            voc.pop(i)
    return rare_words_num

print("word size:" + str(len(word_set)))
print("pos_word vocabulary size:" + str(len(pos_word)))
print("pos_word_num:" + str(pos_word_num))
print("neg_word vocabulary size:" + str(len(neg_word)))
print("neg_word_num:" + str(neg_word_num))

rare_pos_word_num = omit(5,pos_word)
rare_neg_word_num = omit(5,neg_word)
pos_word_num -= rare_pos_word_num
neg_word_num -= rare_neg_word_num
print('After omit:')
print("pos_word vocabulary size:" + str(len(pos_word)))
print("pos_word_num:" + str(pos_word_num))
print("neg_word vocabulary size:" + str(len(neg_word)))
print("neg_word_num:" + str(neg_word_num))
```

    word size:114697
    pos_word vocabulary size:76967
    pos_word_num:2361419
    neg_word vocabulary size:75631
    neg_word_num:2306452
    After omit:
    pos_word vocabulary size:17479
    pos_word_num:2264438
    neg_word vocabulary size:16223
    neg_word_num:2209833
    

c.Calculate the following probability


```python
the_num1 = 0
the_num2 = 0
for sentence in train_pos:
    if 'the' in sentence:
        the_num1 +=1 
for sentence in train_neg:
    if 'the' in sentence:
        the_num2 +=1 
P_the = (the_num1+the_num2) / (train_pos_num+train_neg_num)
print('P[“the”] =' + str (P_the))
P_the2 = the_num1 / train_pos_num
print('P[“the” | Positive]  =' + str (P_the2))
```

    P[“the”] =0.991
    P[“the” | Positive]  =0.9896
    


```python
p_pos = train_pos_num+1.0 / (train_pos_num+train_neg_num+1.0)
p_neg = train_neg_num+1.0 / (train_pos_num+train_neg_num+1.0)

def classify(features): #(train_feature,train_label,test_feature)
    P_pos = math.log(p_pos)
    P_neg = math.log(p_neg)
    for feature in features:
        if feature in pos_word:
            P_feature_in_pos = (float(pos_word[feature])+1) / (float(pos_word_num)+1)
        elif feature not in pos_word:
            P_feature_in_pos = 1 / (float(pos_word_num)+1)
        P_pos += math.log(P_feature_in_pos)
        if feature in neg_word:
            P_feature_in_neg = (float(neg_word[feature])+1) / (float(neg_word_num)+1)
        elif feature not in neg_word:
            P_feature_in_neg = 1 / (float(neg_word_num)+ 1)
        P_neg += math.log(P_feature_in_neg)
    if P_pos > P_neg:
        return 'pos'
    return 'neg'

```

d.Calculate accuracy using dev dataset

e.Do following experiments


```python
kf  = KFold(5, True , 10)
X = dev_pos
sum = 0.0
for train_index , test_index in kf.split(X):
    pos = []
    pos_test = []
    neg = []
    neg_test = []
    for i in train_index:
        pos.append(dev_pos[i])
        neg.append(dev_neg[i])
    for i in test_index:
        pos_test.append(dev_pos[i])
        neg_test.append(dev_neg[i])
    pos_voc,pos_w_num = word_num(pos)
    neg_voc,neg_w_num = word_num(neg)

    p_pos1 = len(pos)+1.0 / (len(pos)+len(neg)+1.0)
    p_neg1 = len(neg)+1.0 / (len(pos)+len(neg)+1.0)
    
    def classify_dev(features): #(train_feature,train_label,test_feature)
        P_pos = math.log(p_pos1)
        P_neg = math.log(p_neg1)
        for feature in features:
            if feature in pos_voc:
                #P_feature_in_pos = (float(pos_voc[feature])+1.0) / (float(pos_w_num)+1.0)
                #P_feature_in_pos = (float(pos_voc[feature])+2.0) / (float(pos_w_num)+2.0)
                P_feature_in_pos = (float(pos_voc[feature])+0.5) / (float(pos_w_num)+0.5)
            elif feature not in pos_voc:
                #P_feature_in_pos = 1.0 / (float(pos_w_num)+1.0)
                #P_feature_in_pos = 2.0 / (float(pos_w_num)+2.0)
                P_feature_in_pos = 0.5 / (float(pos_w_num)+ 0.5)
            P_pos += math.log(P_feature_in_pos)
            if feature in neg_voc:
                #P_feature_in_neg = (float(neg_voc[feature])+1.0) / (float(neg_w_num)+1.0)
                #P_feature_in_neg = (float(neg_voc[feature])+2.0) / (float(neg_w_num)+2.0)
                P_feature_in_neg = (float(neg_voc[feature])+0.5) / (float(neg_w_num)+0.5)
            elif feature not in neg_voc:
                #P_feature_in_neg = 1.0 / (float(neg_w_num)+1.0)
                #P_feature_in_neg = 2.0 / (float(neg_w_num)+2.0)
                P_feature_in_neg = 0.5 / (float(neg_w_num)+0.5)
            P_neg += math.log(P_feature_in_neg)
        if P_pos > P_neg:
            return 'pos'
        return 'neg'
    
    random.shuffle(pos_test)
    random.shuffle(neg_test)
    result1 = []
    result2 = []
    for i in pos_test:
        predict = classify_dev(i)
        result1.append(predict)
    for i in neg_test:
        predict = classify_dev(i)
        result2.append(predict)
    score1 = accuracy_score(['pos']*500,result1)
    score2 = accuracy_score(['neg']*500,result2)
    score = (score1+score2)/2
    sum += score
    print(score)

final_score = sum / 5
print('final_score:'+str(final_score))
```

    0.892
    0.893
    0.921
    0.898
    0.89
    final_score:0.8988000000000002
    

add 0.5 smooth:
0.892
0.89
0.923
0.898
0.89
final_score:0.8986000000000001

add 1 smooth:
0.884
0.884
0.915
0.893
0.886
final_score:0.8924

add 2 smooth:
0.878
0.876
0.905
0.883
0.8779999999999999
final_score:0.884

Derive Top 10 words that predicts positive and negative class


```python
top10_pos = sorted(pos_word.items() , key=lambda pos_word:pos_word[1] ,reverse= True)
print('Top 10 words that predicts positive:\n')
for i in range(10):
    print(top10_pos[i][0])

top10_neg = sorted(neg_word.items() , key=lambda pos_word:pos_word[1] ,reverse= True)
print('\nTop 10 words that predicts negative:\n')
for i in range(10):
    print(top10_neg[i][0])
```

    Top 10 words that predicts positive:
    
    film
    movie
    one
    like
    good
    story
    great
    time
    see
    well
    
    Top 10 words that predicts negative:
    
    movie
    film
    one
    like
    no
    even
    good
    would
    bad
    really
    

f.Using the test dataset


```python
random.shuffle(test_pos)
random.shuffle(test_neg)
result1 = []
result2 = []
for i in test_pos:
    predict = classify(i)
    result1.append(predict)
for i in test_neg:
    predict = classify(i)
    result2.append(predict)
score1 = accuracy_score(['pos']*12500,result1)
score2 = accuracy_score(['neg']*12500,result2)
score = (score1+score2)/2
print('The final accuracy:'+str(score))
```

    The final accuracy:0.80708
    


```python

```
