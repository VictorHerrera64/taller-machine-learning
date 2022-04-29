import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier

simplefilter(action='ignore', category=FutureWarning)
# Leer csv diabetes
path = './dataSets/diabetes/dataSet.csv'
data = pd.read_csv(path)
#Limpiar y normalizar la data
'''
columna Glucose
'''
data.Glucose.replace(np.nan, 120, inplace=True)
rangos = [ 70, 100 ,120, 150, 170, 200]
nombres = ['1', '2', '3', '4', '5']
data.Glucose = pd.cut(data.Glucose, rangos, labels=nombres)
'''
columna Age
'''
rangos = [ 20, 30, 40, 50, 70, 100]
nombres = ['1', '2', '3', '4', '5']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
'''
columna BMI
'''
data.BMI.replace(np.nan, 32, inplace=True)
rangos = [ 10, 20, 30, 40, 50, 70]
nombres = ['1', '2', '3', '4', '5']
data.BMI = pd.cut(data.BMI, rangos, labels=nombres)
'''
columna DiabetesPedigreeFunction
'''
rangos = [ 0.05, 0.25, 0.50, 1, 1.50, 2.50]
nombres = ['1', '2', '3', '4', '5']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rangos, labels=nombres)
'''
columna BloodPressure
'''
rangos = [ 0, 20, 40, 60, 80, 100, 130]
nombres = ['1', '2', '3', '4', '5', '6']
data.BloodPressure = pd.cut(data.BloodPressure, rangos, labels=nombres)
'''
columna SkinThickness
'''
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.SkinThickness = pd.cut(data.SkinThickness, rangos, labels=nombres)
'''
columna Insulin
'''
rangos = [ 0, 100, 200, 300, 400, 500, 700, 900]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Insulin = pd.cut(data.Insulin, rangos, labels=nombres)

#Borrar NaN
data.dropna(axis=0,how='any', inplace=True)
#Dropear los datos
data.drop(['Pregnancies'], axis= 1, inplace = True)
#Partir la tabla en dos
data_train = data[:383]
data_test = data[383:]
x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)
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

