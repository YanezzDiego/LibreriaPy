import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Cor_InfoMutua(df, exportar=False, nombre=False, mostrar_plot=True, fill="upper", colores=["red", "white", "blue"]):
    # Función para calcular la entropía de una variable categórica
    def entropy(x):
        x = pd.Categorical(x)
        probabilidades = x.value_counts() / len(x)
        H = -np.sum(probabilidades * np.log2(probabilidades))
        return H

    # Función para calcular la información mutua entre dos variables categóricas
    def calcular_informacion_mutua(x, y):
        entropia_x = entropy(x)
        entropia_y = entropy(y)
        conjunta_xy = pd.crosstab(x, y) / len(x)
        entropia_conjunta = -np.sum(conjunta_xy * np.log2(conjunta_xy + 1e-10)).sum()
        info_mutua = entropia_x + entropia_y - entropia_conjunta
        return info_mutua

    # Separamos el df en aquellas variables numericas y aquellas categóricas para el calculo de correlación o info mutua respectivamente
    df_numerico = df.select_dtypes(include=[np.number])
    df_categorico = df.select_dtypes(include=['category', 'object'])

    # Inicializamos matrices
    matriz_correlacion = None
    matriz_info_mutua = None

    # Se calculan la matriz de correlaciones
    if df_numerico.shape[1] > 1:
        matriz_correlacion = df_numerico.corr()

    # Se calcula la matriz de información mutua
    if df_categorico.shape[1] > 1:
        matriz_info_mutua = pd.DataFrame(np.nan, index=df_categorico.columns, columns=df_categorico.columns)
        max_info_mutua = 0

        for i in range(df_categorico.shape[1]):
            for j in range(i + 1, df_categorico.shape[1]):
                info_mutua = calcular_informacion_mutua(df_categorico.iloc[:, i], df_categorico.iloc[:, j])
                matriz_info_mutua.iloc[i, j] = info_mutua
                matriz_info_mutua.iloc[j, i] = info_mutua
                max_info_mutua = max(max_info_mutua, info_mutua)

        # Normalizar solo si max_info_mutua es mayor que 0
        if max_info_mutua > 0:
            matriz_info_mutua = matriz_info_mutua / max_info_mutua

    # Volvemos a juntar ambas matrices.
    matriz_final = pd.DataFrame(np.nan, index=df.columns, columns=df.columns)
    if matriz_correlacion is not None:
        matriz_final.update(matriz_correlacion)
    if matriz_info_mutua is not None:
        matriz_final.update(matriz_info_mutua)

    #Si el nombre para exportar el gráfico está señalado, utilizar.
    # Si no está señalado, el nobmre del archivo exportado será "variable" + "_cor_info.pdf"
    if exportar:
        if not nombre:
            nombre_variable = 'df'
            nombre_archivo = f"{nombre_variable}_cor_info.pdf"
        else:
            nombre_archivo = nombre
        plt.savefig(nombre_archivo)

    # Graficar la matriz de correlaciones / info mutua
    if mostrar_plot or exportar:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_final.astype(float), annot=True, cmap=sns.color_palette(colores), mask=(matriz_final.isnull()) if fill == "upper" else None, linewidths=0.5, linecolor='gray', cbar_kws={'shrink': 0.8})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title("Matriz de Correlación e Información Mutua")
        plt.tight_layout()
        plt.show()

    # Retornar ambas matrices en una lista
    return {
        'matriz_correlacion': matriz_correlacion,
        'matriz_info_mutua': matriz_info_mutua
    }