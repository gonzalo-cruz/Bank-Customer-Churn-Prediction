
# Función KNN 
knn_manual <- function(train_data, train_labels, test_data, k) {
  num_test_samples <- nrow(test_data)
  predicciones <- character(num_test_samples)
  
  for (i in 1:num_test_samples) {
    fila_prueba <- test_data[i, ]
    distancias <- numeric(nrow(train_data))
    
    # Calcular la distancia euclídea a cada punto del conjunto de entrenamiento
    for (j in 1:nrow(train_data)) {
      fila_entrenamiento <- train_data[j, ]
      distancias[j] <- sqrt(sum((fila_entrenamiento - fila_prueba)^2))
    }
    
    # Combinar distancias con las etiquetas
    distancia_etiquetas <- data.frame(distancia = distancias, etiqueta = train_labels)
    
    # Ordenar por distancia
    distancia_etiquetas_ordenada <- distancia_etiquetas[order(distancia_etiquetas$distancia), ]
    
    # Obtener las etiquetas de los k vecinos más cercanos
    etiquetas_vecinos_cercanos <- as.character(distancia_etiquetas_ordenada$etiqueta[1:k])
    
    # Predecir la clase (la más frecuente)
    tabla_vecinos <- table(etiquetas_vecinos_cercanos)
    clase_predicha <- names(tabla_vecinos)[which.max(tabla_vecinos)]
    predicciones[i] <- clase_predicha
  }
  return(predicciones)
}

# Ejemplo de uso para KNN con datos aleatorios
set.seed(789)
num_train_knn <- 50
num_test_knn <- 20
num_features_knn <- 2

train_data_aleatorio_knn <- data.frame(
  feature1 = rnorm(num_train_knn),
  feature2 = rnorm(num_train_knn)
)
train_labels_aleatorio_knn <- factor(sample(c("ClaseX", "ClaseY"), num_train_knn, replace = TRUE))

test_data_aleatorio_knn <- data.frame(
  feature1 = rnorm(num_test_knn),
  feature2 = rnorm(num_test_knn)
)

k_value_aleatorio_knn <- 5

predicciones_aleatorias_knn <- knn_manual(
  train_data_aleatorio_knn,
  train_labels_aleatorio_knn,
  test_data_aleatorio_knn,
  k_value_aleatorio_knn
)

cat("Predicciones de KNN para datos de prueba aleatorios (k=", k_value_aleatorio_knn, "):\n")
print(predicciones_aleatorias_knn)
cat("\nPrimeras 10 etiquetas de entrenamiento aleatorias de KNN:\n")
print(head(train_labels_aleatorio_knn, 10))
