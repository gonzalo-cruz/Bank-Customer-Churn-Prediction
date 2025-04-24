# Función de AdaBoost manual

adaboost_manual_test <- function(X, y, n_estimadores = 10) {
  # X: datos entrenamiento
  # y: etiquetas
  # n_estimadores: numero de clasificadores
  n_samples <- nrow(X)
  pesos <- rep(1/n_samples, n_samples) # Inicializamos los pesos de las observaciones
  estimadores <- list()
  pesos_estimadores <- numeric(n_estimadores)
  for (m in 1:n_estimadores) {
    # Ajustamos el clasificador utilizando los pesos
    mejor_stump <- encontrar_mejor_stump(X, y, pesos) 
    estimadores[[m]] <- mejor_stump
    
    # Calculamos el error ponderado del clasificador
    predicciones <- predecir_stump(X, mejor_stump)
    incorrecto <- ifelse(y != predicciones, 1, 0)
    error_ponderado <- sum(pesos * incorrecto) / sum(pesos) # normalizamos
    
    # Calculamos el coeficiente alpha
    # Añadimos un epsilon muy pequeño para evitar problemas con 0s y 1s
    epsilon <- 1e-10
    alpha_m <- 0.5 * log((1 - error_ponderado + epsilon) / (error_ponderado + epsilon))
    pesos_estimadores[m] <- alpha_m
    
    # Actualizamos los pesos de las observaciones
    pesos <- pesos * exp(alpha_m * incorrecto)
    pesos <- pesos / sum(pesos) # normalizamos
  }
  # Devolvemos el modelo final con los pesos ajustados
  return(list(estimators = estimadores, estimator_weights = pesos_estimadores))
}

# Utilizamos stumps (árboles con un solo nivel)
# Funcion para devolver el mejor stump
encontrar_mejor_stump <- function(X, y, pesos) {
  mejor_error <- Inf
  mejor_stump <- NULL
  n_features <- ncol(X)
  n_samples <- nrow(X)
  for (j in 1:n_features) {
    valores_feature <- unique(X[, j])
    for (umbral in valores_feature) {
      for (polaridad in c(1, -1)) {
        predicciones <- ifelse(polaridad * X[, j] > polaridad * umbral, 1, -1)
        incorrecto <- ifelse(predicciones != y, 1, 0)
        error_ponderado <- sum(pesos * incorrecto) / sum(pesos)
        if (error_ponderado < mejor_error) {
          mejor_error <- error_ponderado
          mejor_stump <- list(feature_index = j, threshold = umbral, polarity = polaridad)
        }
      }
    }
  }
  return(mejor_stump)
}

# predecimos el stump
predecir_stump <- function(X, stump) {
  predicciones <- ifelse(stump$polarity * X[, stump$feature_index] > stump$polarity * stump$threshold, 1, -1)
  return(predicciones)
}

# Funcion para la prediccion final del modelo
predecir_adaboost_manual_test <- function(modelo, X) {
  n_estimadores <- length(modelo$estimators)
  predicciones <- matrix(0, nrow = nrow(X), ncol = n_estimadores)
  for (m in 1:n_estimadores) {
    predicciones[, m] <- modelo$estimator_weights[m] * predecir_stump(X, modelo$estimators[[m]])
  }
  predicciones_finales <- sign(rowSums(predicciones))
  return(predicciones_finales)
}

# Ejemplo para probar con datos aleatorios
# Generamos datos aleatorios 
set.seed(123)
n_muestras <- 100
n_features <- 3
X_aleatorio <- as.data.frame(matrix(rnorm(n_muestras * n_features), ncol = n_features))

# Generamos etiquetas aleatorias 
y_aleatorio <- sample(c(-1, 1), n_muestras, replace = TRUE)

# Entrenamos el modelo AdaBoost
modelo_prueba <- adaboost_manual_test(X_aleatorio, y_aleatorio, n_estimadores = 30)

# Hacemos predicciones con el modelo entrenado
predicciones_prueba <- predecir_adaboost_manual_test(modelo_prueba, X_aleatorio)

# Evaluamos la "accuracy" (en este caso es con datos aleatorios, así que no es significativo)
accuracy_prueba <- mean(predicciones_prueba == y_aleatorio)
cat("Accuracy con datos aleatorios:", accuracy_prueba, "\n")

# Imprimimos algunas de las predicciones para ver el resultado
print("Primeras 10 predicciones:")
print(head(predicciones_prueba, 10))
print("Primeras 10 etiquetas reales:")
print(head(y_aleatorio, 10))