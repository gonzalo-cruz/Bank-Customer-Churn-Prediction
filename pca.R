
# Cargamos el dataset
df <- read.csv('Universidad/Primero/2Q/MLI/Bank Customer Churn Prediction.csv')

# Quitamos las columnas que corresponden a variables no continuas
df <- subset(df, select = -c(customer_id, churn, credit_card, products_number, gender, country, active_member))

# funcion pca
pca <- function(...) {
  centered <- scale(df, center = TRUE, scale = FALSE)
  
  cov_matrix <- cov(centered)
  
  eigen_r <- eigen(cov_matrix)
  
  orden <- order(eigen_r$values, decreasing = TRUE)
  eigen_values <- eigen_r$values[orden]
  eigen_vectors <- eigen_r$vectors[, orden]
  
  varianza <- eigen_values / sum(eigen_values)
  
  return(list(
    componentes_principales = eigen_vectors,
    valores_propios = eigen_values,
    varianza_explicada = varianza
  ))
}

# aplicamos el pca
pca_result <- pca(df)

# imprimimos el resultado
print("Principal Components:")
print(pca_result$componentes_principales)

# comparamos con los resultados de la funcion de r
prcomp(df)
