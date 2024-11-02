La librería **EntregaLibreriaPy** contiene 5 funciones que permiten gestionar *data sets*.

- **CalcularMetricas**: Esta función entrega una lista con las métricas de AUC, varianza y entropía de las variables de un *dataframe*. Además, entrega la posibilidad de poder graficar AUC con respecto a la variable y.
- **Correlacion_InfoMutua**: Esta función entrega una matriz de correlaciones/ganancia de información para las variables numéricas y categóricas respectivamente. Además, entrega la opción de poder graficar estas matrices.
- **Discretizacion**: Esta función permite discretizar por los métodos *Equal Frequency* y *Equal Width* las variables de un *dataset*, obteniendo las categorías y los puntos de corte de los intervalos.
- **FiltradoDataSetMetricas**: Esta función permite filtrar un *dataset* por las métricas AUC, varianza y entropía. Requiere que el *dataset* sea completamente numérico o completamente categórico.
- **NormalizacionEstandarizacion**: Esta función permite aplicar técnicas de transformación de datos.
