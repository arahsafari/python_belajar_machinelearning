import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv("datasetsbp.csv")
df_test = pd.read_csv("datatestsbp.csv")

#ini buat nge test akurasi dari data train
#X_train = df.iloc[:112,1:8].apply(preprocessing.LabelEncoder().fit_transform)
#Y_train = df.iloc[:112,8:9].apply(preprocessing.LabelEncoder().fit_transform)
#X_test = df.iloc[112:160,1:8].apply(preprocessing.LabelEncoder().fit_transform)
#Y_test = df.iloc[112:160,8:9].apply(preprocessing.LabelEncoder().fit_transform)

#ini baru buat nge test data testnya
X_train = df.iloc[:,:-1].apply(preprocessing.LabelEncoder().fit_transform)
Y_train = df.iloc[:,5:6].apply(preprocessing.LabelEncoder().fit_transform)
X_test = df_test.iloc[:,:-1].apply(preprocessing.LabelEncoder().fit_transform)
Y_test = df_test.iloc[:,5:6].apply(preprocessing.LabelEncoder().fit_transform)


from sklearn.naive_bayes import *
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred = gnb.predict(X_test)



from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))

# print(y_pred)
# print(Y_test)