import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df_train = pd.read_excel('train.xlsx')
df_test = pd.read_excel('test.xlsx')

df_train['JENIS KELAMIN'] = df_train['JENIS KELAMIN'].apply(lambda x: 1 if x == "LAKI - LAKI" else 0)
df_train['STATUS MAHASISWA'] = df_train['STATUS MAHASISWA'].apply(lambda x: 1 if x == "MAHASISWA" else 0)
df_train['STATUS NIKAH'] = df_train['STATUS NIKAH'].apply(lambda x: 1 if x == "BELUM MENIKAH" else 0)

df_test['JENIS KELAMIN'] = df_test['JENIS KELAMIN'].apply(lambda x: 1 if x == "LAKI - LAKI" else 0)
df_test['STATUS MAHASISWA'] = df_test['STATUS MAHASISWA'].apply(lambda x: 1 if x == "MAHASISWA" else 0)
df_test['STATUS NIKAH'] = df_test['STATUS NIKAH'].apply(lambda x: 1 if x == "BELUM MENIKAH" else 0)


#select independet and dependent variable
X_train = df_train[['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH','UMUR', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8']]
y_train = df_train['STATUS KELULUSAN']

X_test = df_test[['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH', 'UMUR', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8']]
y_test = df_test['STATUS KELULUSAN']


knn_Manhattan = KNeighborsClassifier(n_neighbors = 1, p=1)
knn_Manhattan.fit(X_train, y_train)
knn_Euclidean = KNeighborsClassifier(n_neighbors = 1, p=2)
knn_Euclidean.fit(X_train, y_train)
knn_Minkowski = KNeighborsClassifier(n_neighbors = 1, p=3)
knn_Minkowski.fit(X_train, y_train)

models = {
    'knn_Manhattan' : knn_Manhattan,
    'knn_Euclidean' : knn_Euclidean,
    'knn_Minkowski' : knn_Minkowski
}


with open('knn.pkl', 'wb') as file:
    pickle.dump(models, file)