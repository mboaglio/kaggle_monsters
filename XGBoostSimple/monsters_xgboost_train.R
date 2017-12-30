# XGBoost  con  busqueda bayesiana ... y cross validation
# Para variable continua usando el dataset monsters de Kaggle
#

#limpio la memoria
rm( list=ls() )
gc()
options(na.action='na.pass')
#options(na.action='na.omit')
#options(na.action='na.fail')

library("xgboost")
library("Matrix")
library("rBayesianOptimization" )

setwd( "/Users/marcos/odrive/ITBA/Kaggle/monsters/XGBoostSimple")

archivo_entrada <- "../train.csv"
archivo_grid    <- "monsters_grid.txt"
archivo_salida  <- "monsters_salida.txt"
campos_a_borrar <- c( "Id")
campo_clase     <- "type"

#genero el archivo de salida si no existe
if( !file.exists( archivo_salida) )
{
cat( "fecha", "entrada", "algoritmo", "obs", 
     "subsamble", "colsample_bytree",
     "eta",  "alpha", "lambda", "gamma",
     "min_child_weight", "max_depth", "iteracion", "tiempo", 
     "merror",  
     "\r\n", sep="\t", file=archivo_salida, fill=FALSE, append=FALSE 
    ) 
}


#genera la linea con los nombres de campos en el archivo grid si no existe, es case senstive y  ?Value? es con la V mayuscula
if( !file.exists( archivo_grid) )
{
cat( "peta",               "\t", 
     "palpha",             "\t",
     "plambda",            "\t", 
     "pgamma",             "\t",
     "pmin_child_weight",  "\t", 
     "pmax_depth",         "\t", 
      "Value",     "\r\n", 
      file=archivo_grid, fill=FALSE, append=FALSE 
    ) 
}

init_grid <- read.table( archivo_grid, header=TRUE, sep="\t" )
pinit_points  <- max( 50- length( init_grid[,2] ),    0 )

#leo el dataset
dataset  <- read.table( archivo_entrada, header=TRUE, sep="," )

#borro las variables que no me interesan
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 

# Guardo los labels y los tranformo desde 0 a 2
tlabels <- as.numeric(as.factor(dataset$type))-1
plabels <- levels(as.factor(dataset$type))
numberOfClasses <- max(tlabels) + 1
dataset$type <- as.numeric(factor(dataset$type))-1
dataset$color <- as.numeric(factor(dataset$color))

# Preparo los datos con one-hot-encoding para xgboost
formula <- formula(paste(campo_clase, "~.-1"))
dtrain  <- sparse.model.matrix( formula, data = dataset, drop.unused.levels = FALSE )

#estos parametros estan puestos fijos
vnround            <- 5000
psubsample         <-    1.0
pcolsample_bytree  <-    1.0

#---------------------------------------------------------
xgb_cv_bayes <- function( peta,  palpha, plambda, pgamma, pmin_child_weight, pmax_depth )  {

	set.seed( 599941  )

	t0 =  Sys.time()
	cv.linear    = xgb.cv( 
				data = dtrain,  label=dataset[,campo_clase], # missing = 0 ,
	        	        stratified = TRUE,       nfold = 5 ,
 				subsample = psubsample, 
	 			colsample_bytree = pcolsample_bytree, 
		                eta = peta,
 				min_child_weight = pmin_child_weight, 
	 			max_depth = pmax_depth,
		 		alpha = palpha, lambda = plambda, gamma = pgamma,
	 			objective="multi:softmax",   
 				# eval_metric = "merror", 
				num_class = numberOfClasses, maximize =TRUE,
	      nround= vnround,   early_stopping_rounds = 100
				)

	t1 = Sys.time()

	# Comparo
	tiempo        <- as.numeric(  t1 - t0, units = "secs")

	# Pongo merror en negativo porque BayesOptimization va a MAXIMIZAR la funcion!, abajo lo devuelvo como negativo.
	merror       <- -1 * min( cv.linear$evaluation_log[ , test_merror_mean] )
	iteracion_min <- which.min(  cv.linear$evaluation_log[ , test_merror_mean] )
	
	cat( format(Sys.time(), "%Y%m%d %H%M%S"), archivo_entrada, "xgboost_merror", "todas",
	     psubsample, pcolsample_bytree,
	     peta,  palpha, plambda, pgamma,
	     pmin_child_weight, pmax_depth, 
	     iteracion_min, tiempo, merror,  
	     "\r\n", sep="\t", file=archivo_salida, fill=FALSE, append=TRUE 
	    ) 

	cat(  peta, "\t", palpha, "\t", plambda, "\t", pgamma, "\t", pmin_child_weight, "\t", pmax_depth, "\t", merror, "\r\n",
	      file=archivo_grid, fill=FALSE, append=TRUE 
	    ) 

	# confusion matrix
	#confusionMatrix(factor(OOF_prediction$label), 
	#                factor(OOF_prediction$max_prob),
	#                mode = "everything")
	
	
	
	# Devuelvo Score 
	list( Score = merror, Pred = 0 )

}

#---------------------------------------------------------
OPT_Res <- BayesianOptimization( xgb_cv_bayes,
           	bounds = list(
                                peta               =  c(0.25,   0.35),  
                                palpha             =  c(0,   2), 
                                plambda            =  c(0.8,   1.2),
                                pgamma             =  c(0,  2.0),
                                pmin_child_weight  =  c(0.9,  2.0),
				                        pmax_depth         =  c(6L,   20L)
			      ),
	   init_grid_dt = init_grid, init_points = pinit_points,  n_iter = 1000,
	   acq = "ucb", kappa = 2.576, eps = 0.001,
	   verbose = TRUE
	   )

# Parametros DEFAULT
#vnround            =    5000
#peta               =    0.3
#palpha             =    0
#plambda            =    1
#pgamma             =    0
#pmin_child_weight  =    1
#pcolsample_bytree  =    1
#psubsample         =    1
#pmax_depth         =    6

print( OPT_Res )

#limpio la memoria
#rm( list=ls() )
#gc()

#quit( save="no" )

