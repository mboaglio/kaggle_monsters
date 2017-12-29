# Aplicar  XGBoost  a los datos a predecir

#limpio la memoria
rm( list=ls() )
gc()

library("xgboost")
library("Matrix")


#Parametros de entrada
directorio_trabajo    <- "/Users/marcos/odrive/ITBA/Kaggle/monsters"  
archivo_entrada       <- "train.csv"
archivo_aplicar       <- "test.csv"
archivo_prediccion    <- "result.csv"
archivo_parametros    <- "monsters_salida.txt"

campo_clase           <- "type"
formula               <- formula(paste(campo_clase, "~ ."))
campo_id              <- "id"
campo_tipo            <- "type"

setwd( directorio_trabajo )

# Cargo parametros, busco el mejor conjunto de parametros del archivo de corridas bayesianas. Sino pongo los default
if( file.exists( archivo_parametros) )
{
  dataset_parametros  <- read.table( archivo_parametros, header=TRUE, sep="\t" )
  peta               =    dataset_parametros$eta[which.max(dataset_parametros$merror)]
  palpha             =    dataset_parametros$alpha[which.max(dataset_parametros$merror)]
  plambda            =    dataset_parametros$lambda[which.max(dataset_parametros$merror)]
  pgamma             =    dataset_parametros$gamma[which.max(dataset_parametros$merror)]
  pmin_child_weight  =    dataset_parametros$min_child_weight[which.max(dataset_parametros$merror)]
  pmax_depth         =    dataset_parametros$max_depth[which.max(dataset_parametros$merror)]
  #estos parametros son puestos fijos
  vnround            <- 5000
  psubsample         <-    1.0
  pcolsample_bytree  <-    1.0
  
} else {
  
  # Parametros DEFAULT
  vnround            =    5000
  peta               =    0.3
  palpha             =    0
  plambda            =    1
  pgamma             =    0
  pmin_child_weight  =    1
  pcolsample_bytree  =    1
  psubsample         =    1
  pmax_depth         =    6
  
  # Parametros 
  #vnround            =    5000
  #peta               =    .01
  #palpha             =    0
  #plambda            =    1
  #pgamma             =    0
  #pmin_child_weight  =    1
  #pcolsample_bytree  =    .7
  #psubsample         =    .7
  #pmax_depth         =    3
  
}

# Leo y preparo el dataset de entrenamiento
dataset_train  <- read.table( archivo_entrada, header=TRUE, sep=",", row.names=campo_id )
TIPO            <- 0
dataset_train <- cbind(dataset_train,TIPO)

# Leo y preparo el dataset de test o "aplicar"
dataset_aplicar  <- read.table( archivo_aplicar, header=TRUE, sep=",", row.names=campo_id )
type <- "Unknown"
dataset_aplicar <- cbind(dataset_aplicar,type)
TIPO            <- 1
dataset_aplicar <- cbind(dataset_aplicar,TIPO)


# Junto los datasets en uno solo
dataset_unido <- rbind(dataset_train, dataset_aplicar)

# Guardo los labels y los tranformo desde 0 a 2
tlabels <- as.numeric(as.factor(dataset_train$type))-1
plabels <- levels(as.factor(dataset_train$type))
numberOfClasses <- max(tlabels) + 1
dataset_unido$type <- as.numeric(factor(dataset_unido$type))-1
dataset_unido$color <- as.numeric(factor(dataset_unido$color))


# Calculo desde y hasta donde es cada datastet
test_desde   <- which.max(  dataset_unido[  ,campo_tipo ] )
test_hasta   <- nrow( dataset_unido )
train_desde  <- 1
train_hasta  <- test_desde -1 
                                    

#armo el dataset SPARSE
formula  <- formula(paste(campo_clase, "~ .-1"))
dataset_unido_sparse  <- sparse.model.matrix( formula, data = dataset_unido )

#el dataset para entrenar, que automaticamente descarta los registros con clase nula
dtrain       <- dataset_unido_sparse[ train_desde:train_hasta, ]
vector_clase <- dataset_unido[ train_desde:train_hasta, campo_clase ]

set.seed( 102191  )

modelo = xgboost(data = dtrain, label=vector_clase, # missing = 0, 
                 stratified = TRUE, nfold = 5, 
                 subsample = psubsample, 
                 colsample_bytree = pcolsample_bytree, eta = peta, min_child_weight = pmin_child_weight, max_depth = pmax_depth, alpha = palpha,
                 lambda = plambda, gamma = pgamma, objective="multi:softmax", eval_metric = "merror", num_class = numberOfClasses,
                 maximize =TRUE, nround= vnround,
                 early_stopping_rounds = 100, silent = 0, # seed = 599941, 
                 verbose = TRUE )

#los datos nuevos
daplicar    <- dataset_unido_sparse[ test_desde:test_hasta, ]

#aplico el modelo a los datos nuevos
prediccion  = predict(  modelo, daplicar, missing=0 )

#grabo en un archivo la prediccion
tx <- data.frame( row.names( daplicar ),  prediccion )
colnames( tx )  <-  c( "id", "type" )
tx_new <- tx[ order(  - tx[, "type"] ), ]
tx_new$type <- plabels[tx_new$type+1]

write.table( tx_new, file=archivo_prediccion, row.names=FALSE, col.names=TRUE, quote=FALSE, sep=",", eol = "\n")

#rm( list=ls() )
#gc()

