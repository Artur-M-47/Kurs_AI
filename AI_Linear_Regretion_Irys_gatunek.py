import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.linear_model import LinearRegression


path = r"C:\kursy\ML_w_Pythonie_Podstawy_perceptron_regresja\course_data_files\iris\iris.data"
    
iris = pd.read_csv(path,
                   header = None, 
                   names = ['petal length', 'petal width', 
                            'sepal length', 'sepal width', 'species'])
print(iris.head())
iris.head()

# dane do nauki train
# "X" zwyczajowo dane skoleniowe
# 4 kolumny na podstawie których należy okreslic y 
X = iris.iloc[:,:4]

y = iris.loc[:,'species']

categories = {"Iris-setosa":1, 'Iris-versicolor':2, 'Iris-virginica':3}

y=y.apply(lambda row_value: categories[row_value])

print("X print ",X.head())
#X.head()
print("y print ",y.head())


lr = LinearRegression()
# lr.fit uczymy model na danych
lr.fit(X,y)
# lr.score okreslenie trafnosci modelu
lr.score(X,y)

print('lr.score(X,y) ',lr.score(X,y))

# Dane sprawdzające

iris_1 = [5,    3.5,    1.4,    0.2]
iris_2 = [6.4,  3,      4.5,    1]
iris_3 = [6,    3,      5,      2]
iris_4 = [1,    2,      3,      4]

flowers = [iris_1,  iris_2, iris_3, iris_4]
print("flowers\n",flowers)
spieces_predict = lr.predict(flowers)

print(spieces_predict)

#lista = spieces_predict
#for idx_listy in lista:
#    idx_listy= round(idx_listy)
#     print(idx_listy)

for f,s in zip(flowers, spieces_predict):
    if round(s)==1:
        print('Flower {} is {}'.format(f,"Iris-setosa"))
    if round(s)==2:
        print('Flower {} is {}'.format(f,"Iris-versicolor"))
    if round(s)==3:
        print('Flower {} is {}'.format(f,"Iris-virginica"))
    if round(s)==4:
        print('Flower {} is {}'.format(f,"UNKNOWN"))
                
                
# =============================================================================
# flowers_names =[]
# for indx in range(0,len(spieces_predict)):
#     spieces_predict[indx]= round(spieces_predict[indx])
# 
#     for key,value in categories.items():
#         if (value == round(spieces_predict[indx])):
#             flowers_names.append(key)
#             break;
#     if len(flowers_names)!=indx+1:
#         flowers_names.append("UNKNOWN")
# print(flowers_names)
# =============================================================================



