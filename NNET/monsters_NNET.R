#############################################################
# Kaggle - Monsters
# NNET
# Marcos Boaglio
#

rm( list=ls() )
gc()

# Seteo el directorio de trabajo
setwd("/Users/marcos/odrive/ITBA/Kaggle/monsters/NNET")

# Cargo el dataset armado en el otro script
mydata  <- read.table( "../train.csv", header=TRUE, sep=",", dec = ".", row.names="id") 

# Transformo variables categoricas en numericas
mydata$color <- as.numeric(unlist(mydata$color))

# Particiono la base en dos conjuntos, uno de entrenamiento y uno de testeo.
library(caret)
set.seed(8232335)
conjunto <- createDataPartition(y=mydata$type, p=0.70, list=FALSE)
traindata <- mydata[conjunto,]
testdata <- mydata[-conjunto,]

# Red Neuronal NNET con 50 neuronas en la capa oculta
library(nnet)
modelo.rednnet <- nnet(traindata$type ~ ., data= traindata, size = 100, maxit=3000, MaxNWts=50000, trace=T)
nuevotest<-testdata
nuevotest$type<-NULL
nnet.pred<-predict(modelo.rednnet,nuevotest,type=("class"))

# Matriz de Confusion: Red Neuronal NNET con 50 neuronas en la capa oculta
print(confusionMatrix(nnet.pred, testdata$type))
print(nnet.cfmx.tb <- table(nnet.pred, testdata$type))

# Armo heatmap
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 300)
df <- as.matrix(nnet.cfmx.tb)
colnames(df) = c("Ghoul","Goblin","Ghost")
rownames(df) = c("Ghoul","Goblin","Ghost")
heatmap(t(df)[ncol(df):1,], Rowv=NA, Colv=NA, col = my_palette)

# Armo y Guardo predicciones resultado
data_predict_result  <- read.table( "../test.csv", header=TRUE, sep=",", dec = ".") 
data_predict_result$color <- as.numeric(unlist(data_predict_result$color))
nnet.pred<-predict(modelo.rednnet,data_predict_result,type=("class"))
result<-data.frame(id=data_predict_result$id,type=nnet.pred)
write.csv(result,"result.csv",row.names=F)
