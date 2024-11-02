import numpy as np
import pandas as pd

def filtrar_dataset(data, method="entropy", operator=None, umbral=None, col=None, y=None):
    # Validar el argumento method
    valid_methods = ["entropy", "var", "auc"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    # Validación de data frame: solo numérico o solo categórico
    if method == "auc" and not all([np.issubdtype(data.iloc[:, i].dtype, np.number) for i in range(data.shape[1] - 1)]):
        raise ValueError("Para calcular auc, el df debe ser solo numérico")

    if method == "var" and not all([np.issubdtype(data.iloc[:, i].dtype, np.number) for i in range(data.shape[1])]):
        raise ValueError("Para calcular var, el df debe ser solo numérico")

    if method == "entropy" and any([np.issubdtype(data.iloc[:, i].dtype, np.number) for i in range(data.shape[1])]):
        raise ValueError("Para calcular entropía, el df debe ser solo categórico.")

    # Función Entropía
    def entropy(x):
        x = pd.Categorical(x)
        probabilidades = x.value_counts() / len(x)
        H = -np.sum(probabilidades * np.log2(probabilidades))
        return H

    # Función AUC para variables numéricas
    def auc(df):
        df = df.sort_values(by="X", ascending=False)
        TPR = []
        FPR = []

        for valor_corte in df["X"].unique():
            Y_pred = df["X"] > valor_corte
            TP = np.sum((df["Y"] == True) & (Y_pred == True))
            TN = np.sum((df["Y"] == False) & (Y_pred == False))
            FP = np.sum((df["Y"] == False) & (Y_pred == True))
            FN = np.sum((df["Y"] == True) & (Y_pred == False))

            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

            TPR.append(tpr)
            FPR.append(fpr)

        auc_value = np.sum(np.diff(FPR) * (np.array(TPR)[:-1] + np.array(TPR)[1:]) / 2)
        return auc_value

    # Si el usuario no especifica el nombre de las columnas, se filtra sobre todo el df.
    if col is None:
        col = data.columns.tolist()

    # Validación para AUC: debe existir una variable objetivo
    if method == "auc" and y is None:
        raise ValueError("Para filtrar por AUC se debe especificar la variable objetivo 'y'")

    # Conversión de tipos de datos
    data[col] = data[col].apply(lambda x: pd.Categorical(x) if x.dtype == 'object' else x)

    columnas_seleccionadas = col.copy()  # Todas las columnas sobre las cuales se hace el filtro por defecto

    # Se filtra para cada columna
    for columna in col:
        
        # Cálculo de métrica según método
        if method == "entropy":
            valor_metrica = entropy(data[columna])
        elif method == "var":
            valor_metrica = data[columna].var(skipna=True)
        elif method == "auc":
            valor_metrica = auc(pd.DataFrame({"X": data[columna], "Y": data[y]}))
        else:
            raise ValueError("Método no soportado")

        # Operadores posibles
        if operator is None:
            raise ValueError("Debe especificarse un operador para la comparación\n '<','>','==','<=',''>=")
        elif operator == "<":
            comparacion = valor_metrica < umbral
        elif operator == ">":
            comparacion = valor_metrica > umbral
        elif operator == "==":
            comparacion = valor_metrica == umbral
        elif operator == "<=":
            comparacion = valor_metrica <= umbral
        elif operator == ">=":
            comparacion = valor_metrica >= umbral
        else:
            raise ValueError("Operador no soportado")
        if not comparacion:
            columnas_seleccionadas.remove(columna)

    # Retorna el DF filtrado con solo las columnas seleccionadas
    return data[columnas_seleccionadas]

