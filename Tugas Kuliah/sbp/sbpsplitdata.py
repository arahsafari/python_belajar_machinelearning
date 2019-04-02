# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 01:05:41 2019

@author: Kacangrebus
"""

import pandas as pd
from sklearn import preprocessing


#df = pd.read_csv("dataset/creditscreening.csv",header=None).apply(preprocessing.LabelEncoder().fit_transform)
df = pd.read_csv("datatestsbp.csv",header=None)
data_asli = df

#preprocess
#df.replace({'?': 1}, inplace=True)
#df.fillna(df.mean(), inplace=True)
#del df[0]
#del df[1]
#del df[3]
#del df[4]
del df[5]
#del df[6]
#del df[13]
#
#del df[7]
#del df[10]
#del df[14]

data_drop = df

pd.DataFrame(df).to_csv("datatestsbp1.csv",header=None, index=None)


#pembelahan data test dan train ynya di belakang
test = df.sample(frac=0.2,random_state=0)
Y_test = test.iloc[:,-1]
X_test = test.iloc[:,:-1]

df = df.drop(test.index)
Y_train = df.iloc[:,-1]
X_train = df.iloc[:,:-1]

#pembelahan data test dan train ynya di custom
#test = df.sample(frac=0.2,random_state=0)
#Y_test = test.iloc[:,-1]
#X_test = test.iloc[:,1:10]
#
#df = df.drop(test.index)
#Y_train = df.iloc[:,-1]
#X_train = df.iloc[:,1:10]


#ini buat nge test akurasi dari data train
#X_train = df.iloc[:112,1:8].apply(preprocessing.LabelEncoder().fit_transform)
#Y_train = df.iloc[:112,8:9].apply(preprocessing.LabelEncoder().fit_transform)
#X_test = df.iloc[112:160,1:8].apply(preprocessing.LabelEncoder().fit_transform)
#Y_test = df.iloc[112:160,8:9].apply(preprocessing.LabelEncoder().fit_transform)

#ini baru buat nge test data testnya
#X_train = df.iloc[:,1:8].apply(preprocessing.LabelEncoder().fit_transform)
#Y_train = df.iloc[:,8:9].apply(preprocessing.LabelEncoder().fit_transform)
#X_test = df_test.iloc[:,1:8].apply(preprocessing.LabelEncoder().fit_transform)
#Y_test = df_test.iloc[:,8:9].apply(preprocessing.LabelEncoder().fit_transform)



from sklearn.naive_bayes import *
gnb = GaussianNB()
#gnb = MultinomialNB()
#gnb = BernoulliNB()


gnb.fit(X_train, Y_train)
hasil_prediksi = gnb.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, hasil_prediksi))
print(metrics.classification_report(Y_test, hasil_prediksi))
print(metrics.confusion_matrix(Y_test, hasil_prediksi))

p = gnb.predict_proba(test.iloc[:,2:3])


# print(y_pred)
# print(Y_test)