import numpy as np
import pandas as pd

# Función para discretización
def discretize(data, col=None, num_bins=5, method="equal_width"):
    
    # Función para discretización EF
    def discretizeEF(v, num_bins):
        #Validación para las entradas de la función
        if not np.issubdtype(v.dtype, np.number): #np.issubdtype verifica si el tipo de dato del primer argumento (v) es igual a un número
            raise ValueError("El vector de entrada debe ser numérico.")
        if num_bins < 1:
            raise ValueError("El número de num_bins debe ser mayor que 1") 

        v_ordenado = np.sort(v)
        num_por_intervalo = int(np.ceil(len(v) / num_bins))  # Ceil redondea al entero superior
        quiebres = v_ordenado[::num_por_intervalo] #::num_por_intervalo slicing, [inicio : fin : paso] se recorre de inicio a fin, a ritmo num_por_intervalo
        
        # Añadir -Inf al principio y +Inf al final
        quiebres = np.concatenate(([-np.inf], quiebres, [np.inf]))

        # Discretizar los valores
        categorial_result = pd.cut(v, bins=quiebres, include_lowest=True, right=True)

        # Devolver el resultado como un diccionario que entrega las categorías y lso puntos de quiebre
        return {"categorical": categorial_result, "cuts": quiebres[1:num_bins]}
    
    # Función para discretización EW
    def discretizeEW(v, num_bins):
        if not np.issubdtype(v.dtype, np.number):
            raise ValueError("El vector de entrada debe ser numérico.")
        if num_bins < 1:
            raise ValueError("El número de num_bins debe ser mayor que 1")

        anchura = (v.max() - v.min()) / num_bins
        quiebres = np.concatenate(([-np.inf], np.arange(v.min() + anchura, v.max(), anchura), [np.inf]))
        
        # Discretizar los valores
        categorial_result = pd.cut(v, bins=quiebres, include_lowest=True, right=False)

        # Devolver el resultado como un diccionario que entrega las categorías y lso puntos de quiebre
        return {"categorical": categorial_result, "cuts": quiebres[1:num_bins]}

    # Si data es un vector o una serie de pandas, aplicamos la discretización directamente sobre la variable
    if isinstance(data, (np.ndarray, pd.Series)):
        if method == "equal_freq":
            return discretizeEF(data, num_bins)
        elif method == "equal_width":
            return discretizeEW(data, num_bins)
        else:
            raise ValueError("Método de discretización desconocido")
    
    # Si data es un DF
    elif isinstance(data, pd.DataFrame):
        columns = col if col is not None else data.columns #Si no hay columnas vacias, col toma el valor de las columnas del df
        results = {} #se inicializa una lista para almacenar categorias y valores de corte.

        # Aplicación del método
        for column in columns:
            if np.issubdtype(data[column].dtype, np.number):
                if method == "equal_freq":
                    results[column] = discretizeEF(data[column], num_bins)
                elif method == "equal_width":
                    results[column] = discretizeEW(data[column], num_bins)
                else:
                    raise ValueError("Método de discretización desconocido")
            else:
                print(f"Advertencia: La columna '{column}' no es numérica y se omitirá.\n")
        return results
    
    
    else:
        raise ValueError("El par+ametro 'data' debe ser un vector numérico o un DF de pandas.")