
#*********************************************************************************************************
# R Machine Learning in Parallel with SVM
# ref 
# http://jakemdrew.wordpress.com/2014/03/10/machine-learning-in-parallel-with-support-vector-machines-generalized-linear-models-and-adaptive-boosting/
#*********************************************************************************************************


#-------------------------------------------------------------------------------
#-----------------------Setup Parallel Processing-------------------------------
#-------------------------------------------------------------------------------

#number of bootstrap samples to create
sampleCount <- 8

# Run in parallel on Linux using doMC (uncomment for Linux parallel cluster)
#library(doMC)
#registerDoMC(cores=sampleCount) #<- # of processors / hyperthreads on machine

# Run in parallel on Windows using doSNOW (uncomment for Windows parallel cluster)
library(doSNOW)
cluster<-makeCluster(sampleCount) #<- # of processors / hyperthreads on machine
registerDoSNOW(cluster)

#-------------------------------------------------------------------------------
#-----------------------Create Random Sample Test and Training Data-------------
#-------------------------------------------------------------------------------

# Thanks to the UCI repository Magic Gamma telescope data set
# http://archive.ics.uci.edu/ml/machine-learning-databases/
magicGamma = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", header = F, sep=",")

#create unique id for magicGamma file
magicGamma <- data.frame(id=1:nrow(magicGamma),magicGamma) 

# Take 20% random sample of the data for testing, 80% for training
testIndexes <- sample(1:nrow(magicGamma), size=0.2*nrow(magicGamma))
testData <- magicGamma[testIndexes,]
trainData <- magicGamma[-testIndexes,]

# Create random bootstrap training samples (with replacement) in parallel 
trainSamples <- foreach(i = 1:sampleCount) %dopar% {
  trainData[sample(1:nrow(trainData)
                   , size=0.2*nrow(trainData), replace=TRUE),]
} 

#-------------------------------------------------------------------------------
#-----------------------Time Bootstraping Serial vs. Parallel-------------------
#-------------------------------------------------------------------------------                

#-----------------------svm serial-----------------------
timer <- proc.time()
modelDataSvm <- foreach(i = 1:sampleCount) %do% {
  library(e1071)
  svm(V11 ~ ., trainSamples[[i]][,-1], probability=TRUE
      , cost=10, gamma=0.1)
}                
proc.time() - timer 

#-----------------------svm parallel-----------------------
timer <- proc.time()
modelDataSvm <- foreach(i = 1:sampleCount) %dopar% {
  library(e1071)
  svm(V11 ~ ., trainSamples[[i]][,-1], probability=TRUE
      , cost=10, gamma=0.1)
}                
proc.time() - timer                 



#-------------------------------------------------------------------------------
#-----------------------Create Function To Measure Accuracy---------------------
#-------------------------------------------------------------------------------
accuracy <- function (truth, predicted){
  tTable   <- table(truth,predicted)
  print(tTable)
  tp <- tTable[1,1]
  if(ncol(tTable)>1){ fp <- tTable[1,2] } else { fp <- 0}
  if(nrow(tTable)>1){ fn <- tTable[2,1] } else { fn <- 0}
  if(ncol(tTable)>1 & nrow(tTable)>1){ tn <- tTable[2,2] } else { tn <- 0}
  
  return((tp + tn) / (tp + tn + fp + fn))    
}                

#-------------------------------------------------------------------------------
#-----------------------Benchmark Against All Available Training Data-----------
#-------------------------------------------------------------------------------
#-----------------------svm all-----------------------                
timer <- proc.time()
library(e1071)
svmAll <- svm(V11 ~ ., trainData[,-1], probability=TRUE, cost=10, gamma=0.1)
proc.time() - timer

timer <- proc.time()
svmAllTest <- predict(svmAll, testData[,c(-1 ,-ncol(testData))], probability=TRUE)
proc.time() - timer 

#add predicted class and actual class to test data
svmAllResults <- data.frame(id = testData$id, actualClass = testData$V11
                            ,predictedClass = svmAllTest)

#calculate svm all model accuracy
accuracy(svmAllResults$actualClass,svmAllResults$predictedClass) 



#-------------------------------------------------------------------------------
#-----------------------Create Different Bootstrap Models in Parallel-----------
#-------------------------------------------------------------------------------
#-----------------------svm-----------------------

# Could run tune.svm() in parallelbefore this step if needed to get the best
#   values for the cost and gamma parameters.... (very slow)
timer <- proc.time()
modelDataSvm <- foreach(i = 1:sampleCount) %dopar% {
  library(e1071)
  svm(V11 ~ ., trainSamples[[i]][,-1], probability=TRUE
      , cost=10, gamma=0.1)
}                
proc.time() - timer 


#-------------------------------------------------------------------------------
#-----------------------Predict the Test Data in Parallel Using Each Model------
#-------------------------------------------------------------------------------
#-----------------------svm-----------------------
timer <- proc.time()
predictDataSvm <- foreach(i = 1:sampleCount) %dopar% {
  library(e1071)
  predict(modelDataSvm[[i]], testData[,c(-1 ,-ncol(testData))]
          , probability=TRUE)
}
proc.time() - timer

str(predictDataSvm)

str( predictDataSvm[[1]] )

str(  attr( predictDataSvm[[1]], "probabilities" ) )[,"g"]

aa <- attr( predictDataSvm[[1]], "probabilities" )[,"g"]


a <- matrix( nrow=3804, ncol =8 )
for ( i in 1: sampleCount) {
  a[,i] <- attr( predictDataSvm[[i]], "probabilities" )[,"g"]
}




