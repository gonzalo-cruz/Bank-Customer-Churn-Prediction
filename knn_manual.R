
knn_classify_for_loop <- function(train_data, train_labels, test_data, k) {
  num_test_samples <- nrow(test_data)
  predictions <- character(num_test_samples)
  
  for (i in 1:num_test_samples) {
    test_row <- test_data[i, ]
    distances <- numeric(nrow(train_data))
    
    # Calcular la distancia euclídea a cada punto del conjunto de entrenamiento
    for (j in 1:nrow(train_data)) {
      train_row <- train_data[j, ]
      distances[j] <- sqrt(sum((train_row - test_row)^2))
    }
    
    # Combinar distancias con las etiquetas
    distance_labels <- data.frame(distance = distances, label = train_labels)
    
    # Ordenar por distancia
    sorted_distance_labels <- distance_labels[order(distance_labels$distance), ]
    
    # Obtener las etiquetas de los k vecinos más cercanos
    nearest_neighbors_labels <- as.character(sorted_distance_labels$label[1:k])
    
    # Predecir la clase (la más frecuente)
    table_neighbors <- table(nearest_neighbors_labels)
    predicted_class <- names(table_neighbors)[which.max(table_neighbors)]
    predictions[i] <- predicted_class
  }
  return(predictions)
}

# Ejemplo de uso (los mismos datos de antes):
train_data <- data.frame(
  feature1 = c(1, 1, 2, 2, 3, 3),
  feature2 = c(1, 2, 1, 2, 1, 2)
)
train_labels <- factor(c("A", "A", "B", "B", "A", "B"))

test_data <- data.frame(
  feature1 = c(1.5, 2.5),
  feature2 = c(1.5, 1.8)
)

k_value <- 3

predictions_for_loop <- knn_classify_for_loop(train_data, train_labels, test_data, k_value)
print(predictions_for_loop)
