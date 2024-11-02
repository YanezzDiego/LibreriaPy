if __name__ == "__main__":
    calcular_metricas = CalcularMetricas()
    resultados = calcular_metricas.calcular_metricas(data, clase='clase', include=True, only_categorical=False, plot=True, plot_col='feature1')
    print(resultados)