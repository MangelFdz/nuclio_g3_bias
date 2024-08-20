#Función para un overwiew de un dataframe

def check_df(dataframe, tipo):
  # tipo == 'simple' - Solo muestra volumenes y cabecera
  if tipo == 'simple':
    print("1. ¿CUANTAS FILAS Y COLUMNAS hay en el conjunto de datos?")
    num_filas, num_columnas = dataframe.shape
    print("\tHay {:,} filas y {:,} columnas.".format(num_filas, num_columnas))
    print('\n')

    print("2. ¿Cuáles son las PRIMERAS DOS FILAS del conjunto de datos?")
    display(dataframe.head(2))
    print('\n')

  else:
    print("1. ¿CUANTAS FILAS Y COLUMNAS hay en el conjunto de datos?")
    num_filas, num_columnas = dataframe.shape
    print("\tHay {:,} filas y {:,} columnas.".format(num_filas, num_columnas))
    print('\n')

    print("2. ¿Cuáles son las PRIMERAS DOS FILAS del conjunto de datos?")
    display(dataframe.head(2))
    print('\n')

    print("3. ¿Cuáles son las ULTIMAS DOS FILAS del conjunto de datos?")
    display(dataframe.tail(2))
    print('\n')

    print("4. MUESTRA ALEATORIA de filas del conjunto de datos")
    display(dataframe.sample(5))
    print('\n')

    print("5. ¿Cuáles son las COLUMNAS del conjunto de datos?")
    for i in list(dataframe.columns):
      print('\t - ' + i)
    print('\n')

    print("6. ¿Cuál es el TIPO DE DATOS de cada columna?")
    print(dataframe.dtypes)
    print('\n')

    print("7. ¿Cuántas COLUMNAS hay de cada TIPO DE DATOS?")
    print(dataframe.dtypes.value_counts())
    print('\n')

    print("8. INFO más completa sobre la estructura y el contenido del DataFrame")
    print(dataframe.info())
    print('\n')

    print("9. ¿VALORES UNICOS POR COLUMNA?")
    print(dataframe.nunique())
    print('\n')

    print("10. ¿ESTADISTICAS DESCRIPTIVAS BASICAS DE LAS COLUMNAS NUMÉRICAS?")
    display(dataframe.describe(include = 'all').fillna(''))
    print('\n')

    print("11. ¿VALORES NULOS EN EL DATASET?")
    print(dataframe.isnull().sum().sort_values(ascending = False))
    print('\n')

    print("12. PORCENTAJE DE VALORES NULOS EN CADA COLUMNA?")
    print(round((dataframe.isnull().sum()/len(dataframe)*100), 2).sort_values(ascending = False))
    print('\n')