import pytest
import pandas as pd
import numpy as np
from EntregaLibreriaPy import CalcularMetricas, Cor_InfoMutua, DiscretizePY, FiltradoDataSetMetricas, NormalizacionEstandarizacion


def test_all():
    # DF de ejemplo para probar la función calcular_metricas 
    data = pd.DataFrame({
        'feature1': np.random.rand(100),          # Variable continua
        'feature2': np.random.choice(['A', 'B', 'C'], size=100),  # Variable categórica
        'feature3': np.random.randint(1, 10, size=100),  # Variable continua discreta
        'clase': np.random.choice([0, 1], size=100)      # Clase binaria
    })

    data['feature2'] = pd.Categorical(data['feature2'])  # Se castea feature2 a categórica
    # Prueba de función para obtener varianza, auc y entropia para variables numéricas y categóricas (cuando corresponda)
    resultados = calcular_metricas(data, clase='clase', include=True, only_categorical=False, plot=True, plot_col='feature1')

    # Imprimir los resultados
    print(resultados)
    



    # DF para probar funcion Corr_InfoMutua
    data = {
        'Edad': [25, 32, 47, 51, 62, 23, 34, 45, 29, 38],
        'Ingreso': [75000, 50000, 60000, 80000, 120000, 35000, 45000, 78000, 52000, 61000],
        'Género': ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'M'],
        'Ciudad': ['Donostia', 'Barcelona', 'Albacete', 'Donostia', 'Barcelona', 'Bilbao', 'Almeria', 'Barcelona', 'Madrid', 'Eibar']
    }

    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Casteo de variables
    df['Género'] = df['Género'].astype('category')
    df['Ciudad'] = df['Ciudad'].astype('category')
    
    resultado = Cor_InfoMutua(df, mostrar_plot=True)

    # Mostrar los resultados
    print("Matriz de Correlación:")
    print(resultado['matriz_correlacion'])
    print("\nMatriz de Información Mutua:")
    print(resultado['matriz_info_mutua'])
    
    
    
    
    
    
    #Df para probar estandarizacion y normalizacion
    data = {
    'Altura': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Peso': [50, 60, 70, 80, 90, 55, 65, 75, 85, 95]
    }
    df_transform = pd.DataFrame(data_transform)
    
    #Normalizacion
    df_normalizado = transform_data(df_transform.copy(), method="normalizacion")
    print("\nDataFrame Normalizado:")
    print(df_normalizado)

    #Estandarizacion
    df_estandarizado = transform_data(df_transform.copy(), method="estandarizacion")
    print("\nDataFrame Estandarizado:")
    print(df_estandarizado)
    
    
    
    
    #DF para probar la funcion discretize
    data = {
    'Nombre': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10'],
    'Edad': [23, 45, 35, 25, 40, 50, 60, 30, 28, 55],
    'Salario': [3000, 4000, 3500, 2800, 4500, 5000, 5200, 3200, 3100, 4900]
    }
    df = pd.DataFrame(data)

    # Discretizar la columna 'Edad' con igual anchura
    discretized_edad_ew = discretize(df, col=['Edad'], num_bins=3, method="equal_width")

    # Mostrar los resultados de la discretización por igual anchura
    print("Discretización de igual anchura (Edad):")
    print(discretized_edad_ew['Edad']['categorical'])
    print("Puntos de corte:", discretized_edad_ew['Edad']['cuts'])

    # Discretizar la columna 'Edad' con igual frecuencia
    discretized_edad_ef = discretize(df, col=['Edad'], num_bins=3, method="equal_freq")

    # Mostrar los resultados de la discretización por igual frecuencia
    print("\nDiscretización de igual frecuencia (Edad):")
    print(discretized_edad_ef['Edad']['categorical'])
    print("Puntos de corte:", discretized_edad_ef['Edad']['cuts'])
    
    
    
    
    
    
    
    #DF para probar funcion filtrar_dataset
    data = pd.DataFrame({
    'Feature1': np.random.choice(['A', 'B', 'C'], size=100),  # Categórica
    'Feature2': np.random.normal(0, 1, size=100),             # Numérica
    'Feature3': np.random.normal(5, 2, size=100),             # Numérica
    'Target': np.random.choice([True, False], size=100)       # Variable objetivo binaria
    })
    
    # Función para filtrar el DataFrame basada en varianza
    df_filtrado_var = filtrar_dataset(data, method="var", operator=">", umbral=1.0, col=['Feature2', 'Feature3'])
    print("DataFrame filtrado basado en varianza mayor a 1.0:")
    print(df_filtrado_var.head())

    # Función para filtrar el DataFrame basado en entropía
    df_filtrado_entropy = filtrar_dataset(data, method="entropy", operator=">", umbral=1.0, col=['Feature1'])
    print("DataFrame filtrado basado en entropía mayor a 1.0:")
    print(df_filtrado_entropy.head())

    # Función para filtrar el DataFrame basado en AUC
    df_filtrado_auc = filtrar_dataset(data, method="auc", operator=">", umbral=0.5, col=['Feature2', 'Feature3'], y='Target')
    print("DataFrame filtrado basado en AUC mayor a 0.5:")
    print(df_filtrado_auc.head())
