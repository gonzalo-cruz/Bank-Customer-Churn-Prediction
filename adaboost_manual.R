# Función de AdaBoost manual

adaboost_manual <- function(X, y, n = 10) {
  # X: datos entrenamiento
  # y: etiquetas
  # n: numero de casificadores
  n_samples <- nrow(X)
  weights <- rep(1/n_samples, n_samples) # Inicializamos los pesos de las observaciones
  estimators <- list()
  estimator_weights <- numeric(n)
  for (m in 1:n) {
    # Ajustamos el clasificador utilizando los pesos 
    best_stump <- mejor_stump(X, y, weights) # Our weak learner is still a stump
    estimators[[m]] <- best_stump
    
    # Calculamos el error ponderado del clasificador 
    predicciones <- predict_stump(X, best_stump)
    incorrect <- ifelse(y != predicciones, 1, 0)
    weighted_error <- sum(weights * incorrect) / sum(weights) # normalizamos
    
    # Calculamos el coeficiente alpha
    # Añadimos un epsilon muy pequeño para evitar problemas con 0s y 1s
    epsilon <- 1e-10
    alpha_m <- 0.5 * log((1 - weighted_error + epsilon) / (weighted_error + epsilon))
    estimator_weights[m] <- alpha_m
    
    # Actualizamos los pesos de las observaciones
    weights <- weights * exp(alpha_m * incorrect) # Note the change here based on pseudocode
    weights <- weights / sum(weights) # Normalize weights
  }
  # Devolvemos el modelo final con los pesos ajustados
  return(list(estimators = estimators, estimator_weights = estimator_weights))
}

# Utilizamos stumps (árboles con un solo nivel) 
# Funcion para devolver el mejor stump
mejor_stump <- function(X, y, weights) {
  best_error <- Inf
  best_stump <- NULL
  n_features <- ncol(X)
  n_samples <- nrow(X)
  
  for (j in 1:n_features) {
    feature_values <- unique(X[, j])
    for (threshold in feature_values) {
      for (polarity in c(1, -1)) {
        predicciones <- ifelse(polarity * X[, j] > polarity * threshold, 1, -1)
        incorrect <- ifelse(predicciones != y, 1, 0)
        weighted_error <- sum(weights * incorrect) / sum(weights)
        if (weighted_error < best_error) {
          best_error <- weighted_error
          best_stump <- list(feature_index = j, threshold = threshold, polarity = polarity)
        }
      }
    }
  }
  return(best_stump)
}

# predecimos el stump
predict_stump <- function(X, stump) {
  predicciones <- ifelse(stump$polarity * X[, stump$feature_index] > stump$polarity * stump$threshold, 1, -1)
  return(predicciones)
}

# Funcion para la prediccion final del modelo
predict_adaboost_manual <- function(model, X) {
  n <- length(model$estimators)
  predicciones <- matrix(0, nrow = nrow(X), ncol = n)
  for (m in 1:n) {
    predicciones[, m] <- model$estimator_weights[m] * predict_stump(X, model$estimators[[m]])
  }
  predicciones_finales <- sign(rowSums(predicciones))
  return(predicciones_finales)
}

# Ejemplo
# Generamos datos aleatorios
set.seed(42)
X <- as.data.frame(matrix(rnorm(100 * 2), ncol = 2))
y <- ifelse(X$V1^2 + X$V2^2 > 1, 1, -1)

# Entrenamos el modelo
modelo_manual <- adaboost_manual(X, y, n = 50)

# Hacemos predicciones 
predicciones_manual <- predict_adaboost_manual(modelo_manual, X)

# Evaluamos la accuracy
accuracy <- mean(predicciones_manual == y)
cat("Accuracy (Pseudocode-aligned):", accuracy, "\n")
