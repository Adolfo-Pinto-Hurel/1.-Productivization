
# Se importan las librerías necesarias para el problema

import pandas as pd
import numpy as np

# import missingno as msng
import warnings
warnings.filterwarnings('ignore')

# INGESTA Y CONSTRUCCION DE DATOS

df = pd.read_csv('dataset_SCL.csv')

from datetime import datetime

def temporada_alta(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0

df['temporada_alta'] = df['Fecha-I'].apply(temporada_alta)

def dif_min(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    dif_min = ((fecha_o - fecha_i).total_seconds())/60
    return dif_min

df['dif_min'] = df.apply(dif_min, axis = 1)

df['atraso_15'] = np.where(df['dif_min'] > 15, 1, 0)


def get_periodo_dia(fecha):
    fecha_time = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S').time()
    mañana_min = datetime.strptime("05:00", '%H:%M').time()
    mañana_max = datetime.strptime("11:59", '%H:%M').time()
    tarde_min = datetime.strptime("12:00", '%H:%M').time()
    tarde_max = datetime.strptime("18:59", '%H:%M').time()
    noche_min1 = datetime.strptime("19:00", '%H:%M').time()
    noche_max1 = datetime.strptime("23:59", '%H:%M').time()
    noche_min2 = datetime.strptime("00:00", '%H:%M').time()
    noche_max2 = datetime.strptime("4:59", '%H:%M').time()

    if (fecha_time > mañana_min and fecha_time < mañana_max):
        return 'mañana'
    elif (fecha_time > tarde_min and fecha_time < tarde_max):
        return 'tarde'
    elif ((fecha_time > noche_min1 and fecha_time < noche_max1) or
          (fecha_time > noche_min2 and fecha_time < noche_max2)):
        return 'noche'


df['periodo_dia'] = df['Fecha-I'].apply(get_periodo_dia)


# FIN INGESTA Y CONSTRUCCION DE DATOS

# CONSTRUCCION DE:
# x_train
# y_train

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

data = shuffle(df[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'atraso_15']], random_state = 111)

features = pd.concat([pd.get_dummies(data['OPERA'], prefix = 'OPERA'),pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), pd.get_dummies(data['MES'], prefix = 'MES')], axis = 1)
label = data['atraso_15']

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.33, random_state = 42)

# API REST

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Creando una instancia de FastAPI
app = FastAPI()

# Creando una clase que define el request body
# y los tipos de dato de cada atributo
class request_body(BaseModel):
    OPERA_Aerolineas_Argentinas: int
    OPERA_Aeromexico: int
    OPERA_Air_Canada: int
    OPERA_Air_France: int
    OPERA_Alitalia: int
    OPERA_American_Airlines: int
    OPERA_Austral: int
    OPERA_Avianca: int
    OPERA_British_Airways: int
    OPERA_Copa_Air: int
    OPERA_Delta_Air: int
    OPERA_Gol_Trans: int
    OPERA_Grupo_LATAM: int
    OPERA_Iberia: int
    OPERA_JetSmart_SPA: int
    OPERA_KLM: int
    OPERA_Lacsa: int
    OPERA_Latin_American_Wings: int
    OPERA_Oceanair_Linhas_Aereas: int
    OPERA_Plus_Ultra_Lineas_Aereas: int
    OPERA_Qantas_Airways: int
    OPERA_Sky_Airline: int
    OPERA_United_Airlines: int
    TIPOVUELO_I: int
    TIPOVUELO_N: int
    MES_1: int
    MES_2: int
    MES_3: int
    MES_4: int
    MES_5: int
    MES_6: int
    MES_7: int
    MES_8: int
    MES_9: int
    MES_10: int
    MES_11: int
    MES_12: int


# Transformación a numpy array
X = x_train.to_numpy()
Y = y_train.to_numpy()

# Creando y ajustando el modelo, según cambios de formato y tipo.
weights = 'balanced'
logReg_B = LogisticRegression(class_weight=weights)
logReg_B.fit(X, Y)


# Creando un Endpoint para recibir la data
# con la que se hara la predicción.
@app.post('/predict')
def predict(data: request_body):
    # Haciendo que la data este en la forma adecuada para la predicción
    test_data = [[
        data.OPERA_Aerolineas_Argentinas,
        data.OPERA_Aeromexico,
        data.OPERA_Air_Canada,
        data.OPERA_Air_France,
        data.OPERA_Alitalia,
        data.OPERA_American_Airlines,
        data.OPERA_Austral,
        data.OPERA_Avianca,
        data.OPERA_British_Airways,
        data.OPERA_Copa_Air,
        data.OPERA_Delta_Air,
        data.OPERA_Gol_Trans,
        data.OPERA_Grupo_LATAM,
        data.OPERA_Iberia,
        data.OPERA_JetSmart_SPA,
        data.OPERA_KLM,
        data.OPERA_Lacsa,
        data.OPERA_Latin_American_Wings,
        data.OPERA_Oceanair_Linhas_Aereas,
        data.OPERA_Plus_Ultra_Lineas_Aereas,
        data.OPERA_Qantas_Airways,
        data.OPERA_Sky_Airline,
        data.OPERA_United_Airlines,
        data.TIPOVUELO_I,
        data.TIPOVUELO_N,
        data.MES_1,
        data.MES_2,
        data.MES_3,
        data.MES_4,
        data.MES_5,
        data.MES_6,
        data.MES_7,
        data.MES_8,
        data.MES_9,
        data.MES_10,
        data.MES_11,
        data.MES_12
    ]]

    # Prediciendo la clase

    print ("Registro de atributos:")
    print (test_data)
    
    clase = logReg_B.predict(test_data)[0]
    print ("Este registro corresponden a la clase: ", clase)
      
    # Cambiamos tipo de la clase para el return
    list1 = clase.tolist()
    
    # Devuelve el resultado
    return {'El registro de atributos ingresado corresponden a la clase: ': list1}
