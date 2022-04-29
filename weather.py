import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier

simplefilter(action='ignore', category=FutureWarning)
# Leer csv weather
path = './dataSets/weather/dataSet.csv'
data = pd.read_csv(path)
#Limpiar y normalizar la data
'''
Drops de columnas
'''
data.drop(['Date','Location','Rainfall','Evaporation','Sunshine',
'WindGustDir','WindDir9am', 
'WindDir3pm','RISK_MM'], axis= 1, inplace = True)
'''
columna Min Temperatura
'''
rangos = [ -8, 0, 10, 20, 35]
nombres = ['1', '2', '3', '4']
data.MinTemp = pd.cut(data.MinTemp, rangos, labels=nombres)
'''
columna Max Temperatura
'''
rangos = [ -5, 10, 20, 30, 40, 50]
nombres = ['1', '2', '3', '4', '5']
data.MaxTemp = pd.cut(data.MaxTemp, rangos, labels=nombres)
'''
columna WinGustSpeed
'''
data.WindGustSpeed.replace(np.nan, 39, inplace=True)
'''
columna WindSpeed9am
'''
rangos = [ 1, 26, 52 ,78, 94, 110, 130]
nombres = ['1', '2', '3', '4', '5', '6']
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangos, labels=nombres)
'''
columna WindSpeed3pm
'''
rangos = [ 1, 17, 34, 52, 69, 87]
nombres = ['1', '2', '3', '4', '5']
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangos, labels=nombres)
'''
columna Humidity9am 
'''
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity9am = pd.cut(data.Humidity9am, rangos, labels=nombres)
'''
columna Humidity3pm
'''
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity3pm = pd.cut(data.Humidity3pm, rangos, labels=nombres)
'''
columna Pressure9am
'''
rangos = [ 980, 994, 1008, 1022, 1036, 1050]
nombres = ['1', '2', '3', '4', '5']
data.Pressure9am = pd.cut(data.Pressure9am, rangos, labels=nombres)
'''
columna Pressure3pm
'''
rangos = [ 970, 984, 998, 1012, 1026, 1040]
nombres = ['1', '2', '3', '4', '5']
data.Pressure3pm = pd.cut(data.Pressure3pm, rangos, labels=nombres)
'''
columna Cloud9am
'''
data.Cloud9am.replace(np.nan, 4, inplace=True)
rangos = [ 0, 1, 2, 3, 4, 5, 6, 7, 9]
nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.Cloud9am = pd.cut(data.Cloud9am, rangos, labels=nombres)
'''
columna Cloud3pm
'''
data.Cloud3pm.replace(np.nan, 5, inplace=True)
rangos = [ 0, 1, 2, 3, 4, 5, 6, 7, 9]
nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.Cloud3pm = pd.cut(data.Cloud3pm, rangos, labels=nombres)

'''
columna Temp9am
'''
rangos = [ -8, 0, 10, 20, 30, 42]
nombres = ['1', '2', '3', '4', '5']
data.Temp9am = pd.cut(data.Temp9am, rangos, labels=nombres)
'''
columna Temp3pm
'''
rangos = [ -6, 5, 15, 25, 40, 50]
nombres = ['1', '2', '3', '4', '5']
data.Temp3pm = pd.cut(data.Temp3pm, rangos, labels=nombres)

'''
columna RainToday
'''
data.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)
'''
columna RainTomorrow
'''
data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)

data.dropna(axis=0,how='any', inplace=True)

#Partir la tabla en dos
data_train = data[:10000]
data_test = data[10000:]
x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)
# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)
# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')



# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

