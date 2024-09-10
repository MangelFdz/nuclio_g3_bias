import pandas as pd
import numpy as np
import seaborn as sns
import random
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re

def check_df(df_compas_raw, tipo):
  # tipo == 'simple' - Solo muestra volumenes y cabecera
  if tipo == 'simple':
    print("¿Cuántas filas y columnas hay en el conjunto de datos?")
    num_filas, num_columnas = df_compas_raw.shape
    print("\tHay {:,} filas y {:,} columnas.".format(num_filas, num_columnas))
    print('\n########################################################################################')

    print("¿Cuáles son las primeras dos filas del conjunto de datos?")
    display(df_compas_raw.head(2))
    print('\n########################################################################################')

  else:
    print("¿Cuántas filas y columnas hay en el conjunto de datos?")
    num_filas, num_columnas = df_compas_raw.shape
    print("\tHay {:,} filas y {:,} columnas.".format(num_filas, num_columnas))
    print('\n########################################################################################')

    print("¿Cuáles son las primeras dos filas del conjunto de datos?")
    display(df_compas_raw.head(2))
    print('\n########################################################################################')

    print("¿Cuáles son las últimas dos filas del conjunto de datos?")
    display(df_compas_raw.tail(2))
    print('\n########################################################################################')

    print("¿Cómo puedes obtener una muestra aleatoria de filas del conjunto de datos?")
    display(df_compas_raw.sample(2))
    print('\n########################################################################################')

    print("¿Cuáles son las columnas del conjunto de datos?")
    for i in list(df_compas_raw.columns):
      print('\t - ' + i)
    print('\n########################################################################################')

    print("¿Cuál es el tipo de datos de cada columna?")
    print(df_compas_raw.dtypes)
    print('\n########################################################################################')

    print("¿Cuántas columnas hay de cada tipo de datos?")
    print(df_compas_raw.dtypes.value_counts())
    print('\n########################################################################################')

    print("¿Cómo podríamos obtener información más completa sobre la estructura y el contenido del DataFrame?")
    print(df_compas_raw.info())
    print('\n########################################################################################')

    print("¿Cuántos valores únicos tiene cada columna?")
    print(df_compas_raw.nunique())
    print('\n########################################################################################')

    print("¿Cuáles son las estadísticas descriptivas básicas de las columnas numéricas?")
    display(df_compas_raw.describe(include = 'all').fillna(''))
    print('\n########################################################################################')

    print("¿Hay valores nulos en el conjunto de datos?")
    print(df_compas_raw.isnull().sum().sort_values(ascending = False))
    print('\n########################################################################################')
    
    print("Numero de filas duplicadas:")
    print(df_compas_raw.duplicated().sum())
    print('\n########################################################################################')

    print("¿Cuál es el porcentaje de valores nulos en cada columna?")
    print(round((df_compas_raw.isnull().sum()/len(df_compas_raw)*100), 2).sort_values(ascending = False))
    print('\n########################################################################################')

# Función para limpiar y convertir fechas
def limpiar_fecha(fecha):
    if pd.notnull(fecha):
        # Convertir la fecha a string y eliminar la parte de la hora si existe
        fecha = str(fecha).split(' ')[0]  # Solo toma la fecha (parte anterior al espacio)
        # Intentar convertir la fecha al formato datetime
        try:
            return pd.to_datetime(fecha, errors='coerce')
        except:
            return pd.NaT
    return pd.NaT

def procesar_fecha(fecha):
  '''
    * Separados por "-":
      - Patrón 1: 04-01-2020
      - Patrón 2: 2020-01-10
      - Patrón 3: 01-14-20

    * Separados por "/":
      - Patrón 4: 11/01/2020
      - Patrón 5: 02/03/20
  '''

  # Separador '-'

  # %d-%m-%y'
  patron1 = r'^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{2})$'
  # dia: (0[1-9]|[12][0-9]|3[01])
  # mes: (0[1-9]|1[0-2])
  # año: (\d{2})

  #'%d-%m-%Y'
  patron2 = r'^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})$'

  #'%m-%d-%y'
  patron3 = r'^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-(\d{2})$'

  #'%m-%d-%Y'
  patron4 = r'^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-(\d{4})$'

  #'%Y-%m-%d'
  patron5 = r'^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$'

  # Separador '/'

  #'%d/%m/%y'
  patron6 = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{2})$'

  #'%m/%d/%y'
  patron7 = r'^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{2})$'

  #'%m/%d/%Y'
  patron8 = r'^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})$'

  #'%Y/%m/%d'
  patron9 = r'^(\d{4})/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])$'

  #'%Y/%m/%d'
  patron10 = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{4})$'

  # 12/5/2021	
  #'%Y/%m/%d'
  patron11 = r'^(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(\d{4})$'

  # Comprueba si la fecha cumple con el patrón
  if pd.notnull(fecha) and re.fullmatch(patron1, fecha):
    # Parsea la fecha al formato deseado y devuelve en formato "aaaa-mm-dd"
    return pd.to_datetime(fecha, format='%d-%m-%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron2, fecha):
    return pd.to_datetime(fecha, format='%d-%m-%Y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron3, fecha):
    return pd.to_datetime(fecha, format='%m-%d-%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron4, fecha):
    return pd.to_datetime(fecha, format='%m-%d-%Y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron5, fecha):
    return pd.to_datetime(fecha, format='%Y-%m-%d').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron6, fecha):
    return pd.to_datetime(fecha, format='%d/%m/%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron7, fecha):
      return pd.to_datetime(fecha, format='%m/%d/%y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron8, fecha):
      return pd.to_datetime(fecha, format='%m/%d/%Y').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron9, fecha):
      return pd.to_datetime(fecha, format='%Y/%m/%d').strftime('%Y-%m-%d')

  elif pd.notnull(fecha) and re.fullmatch(patron10, fecha):
      return pd.to_datetime(fecha, format='%d/%m/%Y').strftime('%Y-%m-%d')
  
  # 12/5/2021
  elif pd.notnull(fecha) and re.fullmatch(patron11, fecha):
      return pd.to_datetime(fecha, format='%m/%d/%Y').strftime('%Y-%m-%d')

  else:
      # Devuelve la fecha original si no cumple con el patrón o es NaN
      return pd.NaT  # Retorna Not a Time para fechas que no coinciden con ningún formato