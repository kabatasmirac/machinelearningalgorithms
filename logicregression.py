# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)

print(y_pred_knn)

cm_knn=confusion_matrix(y_test,y_pred_knn)
print(cm_knn)



#komşu sayısını değiştirdik ve başarı arttı
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)
cm_knn=confusion_matrix(y_test,y_pred_knn)
print(cm_knn)



from sklearn.svm import SVC
scv = SVC(kernel='linear')
scv.fit(X_train,y_train)
y_pred_scv = scv.predict(X_test)
print("scv")
cm = confusion_matrix(y_test,y_pred_scv)
print(cm)




















