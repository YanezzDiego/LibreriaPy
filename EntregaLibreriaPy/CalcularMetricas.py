import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calcular_metricas(data, clase=None, include=True, only_categorical=False, plot=False, plot_col=None):
    
    # Entropia
    def entropy(x):
        x = pd.Categorical(x)
        probabilidades = pd.Series(x).value_counts() / len(x) #Se castea como un pd. series porque value_counts funciona sobre obj tipo pandas
        H = -np.sum(probabilidades * np.log2(probabilidades))
        return H

    # Función para calcular ROC, AUC y graficar si es que plot = True y la columna coincide con plot_col
    def roc_auc(x, y, should_plot, var_name):
        ordenado = np.argsort(-x) #argsort permite conservar el vinculo con el df luego de ordenar de forma decreciente
        x = x[ordenado]
        y = y[ordenado]

        TPR = []
        FPR = []
        for valor_corte in np.unique(x):
            Y_pred = x > valor_corte
            TP = np.sum((y == 1) & (Y_pred == True))
            TN = np.sum((y == 0) & (Y_pred == False))
            FP = np.sum((y == 0) & (Y_pred == True))
            FN = np.sum((y == 1) & (Y_pred == False))

            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

            TPR.append(tpr)
            FPR.append(fpr)

        # Cálculo de AUC usando el método del trapecio
        auc = np.sum(np.diff(FPR) * (np.array(TPR[:-1]) + np.array(TPR[1:])) / 2)

        # Graficar la curva ROC si should_plot = True
        if should_plot:
            plt.plot(FPR, TPR, color='blue')
            plt.xlabel('FPR (False Positive Rate)')
            plt.ylabel('TPR (True Positive Rate)')
            plt.title(f'Curva ROC para {var_name} - AUC: {round(auc, 2)}')
            plt.show()

        return auc

    # Convertir la variable dependiente a formato binario si es categórica
    if clase is not None and isinstance(data[clase].dtype, pd.CategoricalDtype):
        data[clase] = data[clase].cat.codes #cat codes convierte variables categóricas a numericas

    # Diccionario para almacenar resultados (lista que tendrá auc, varianza y entropia de las varaibles)
    resultados = {}

    # Iterar sobre cada columna del dataset
    for col in data.columns:
        # Calcular entropía si es una variable categórica
        if only_categorical and (isinstance(data[col].dtype, pd.CategoricalDtype) or data[col].dtype == 'object'):
            ent = entropy(data[col])
            resultados[col] = {'entropia': ent}

        elif not only_categorical:
            if pd.api.types.is_numeric_dtype(data[col]) and not isinstance(data[col].dtype, pd.CategoricalDtype):
                # Calcular varianza para variables numéricas
                varianza = np.var(data[col].astype(float), ddof=1)
                resultados[col] = {'varianza': varianza}

                # Calcular AUC si hay una variable dependiente (y) especificada
                if clase is not None and clase in data.columns and len(data[clase].unique()) == 2:
                    # Verificar si se debe graficar esta columna
                    should_plot = plot and (plot_col is not None) and (col == plot_col)
                    auc = roc_auc(data[col].values, data[clase].values, should_plot, var_name=col)
                    resultados[col]['AUC'] = auc

            elif include and (isinstance(data[col].dtype, pd.CategoricalDtype) or data[col].dtype == 'object'):
                # Calcular entropía para variables categóricas solo si include = True
                ent = entropy(data[col])
                resultados[col] = {'entropia': ent}

    return resultados

