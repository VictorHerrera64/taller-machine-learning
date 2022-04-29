import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier

simplefilter(action='ignore', category=FutureWarning)
# Leer csv bank marketing
path = './dataSets/bank-marketing/dataSet.csv'
data = pd.read_csv(path)
#Limpiar y normalizar la data
'''
columna age
'''
rangos = [20, 30, 40, 50, 60, 70, 80, 90]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
'''
Columna job
management 0
technician 1
entrepreneur 2
blue-collar 3
unknown 4
retired 5
'''
data.job.replace(['admin.','unknown','unemployed',
'management','housemaid','entrepreneur',
'student', 'blue-collar','self-employed',
'retired','technician','services'
], [0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
'''
Columna marital
married 0
single 1
divorced 2
'''
data.marital.replace(['married', 'single','divorced'], 
[0,1,2], inplace=True)
'''
Columna education
unknown 0
secondary 1
primary 2
tertiary 3
'''
data.education.replace(['unknown','secondary','primary',
'tertiary'], [0,1,2,3], inplace=True)
'''
Columna default
yes 0
no 1
'''
data.default.replace(['yes','no'], [0,1], inplace=True)
'''
Columna housing
yes 0
no 1
'''
data.housing.replace(['yes','no'], [0,1], inplace=True)
'''
Columna loan
yes 0
no 1
'''
data.loan.replace(['yes','no'], [0,1], inplace=True)
'''
columna balance,contact,day,month,
duration,pdays,previous,campaign(drop)
'''
data.drop(['balance','contact', 'day', 'month',
 'duration','pdays','previous','campaign'], axis= 1, inplace = True)
'''
Columna poutcome
unknown 0
other 1
failure 2
success 3
'''
data.poutcome.replace(['unknown', 'other','failure',
'success'], [0,1,2,3], inplace=True)
'''
Columna y
yes 0
no 1
'''
data.y.replace(['yes','no'], [0,1], inplace=True)
'''
NaN
'''
data.dropna(axis=0,how='any', inplace=True)
#Partir la tabla en dos
data_train = data[:22605]
data_test = data[22605:]
x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)
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











