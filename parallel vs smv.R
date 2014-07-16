#*********************************************************************************************************
# ?? 2013 jakemdrew.com. All rights reserved. 
# This source code is licensed under The GNU General Public License (GPLv3):  
# http://opensource.org/licenses/gpl-3.0.html
#*********************************************************************************************************

#*********************************************************************************************************
# R Machine Learning in Parallel with GLM, SVM, and AdaBoost 
# Created By - Jake Drew
# Version -    1.0, 03/07/2014
#*********************************************************************************************************

#-------------------------------------------------------------------------------
#-----------------------Create Test and Training Data Bootstraps----------------
#-------------------------------------------------------------------------------
sampleCount <- 20

# Thanks to the UCI repository Magic Gamma telescope data set
# http://archive.ics.uci.edu/ml/machine-learning-databases/
magicGamma = read.table(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
  , header = F, sep=",")

summary(magicGamma)

#create unique id for magicGamma file (used later)
magicGamma <- data.frame(id=1:nrow(magicGamma),magicGamma) 

# Take 20% random sample of the data for testing, 80% for training
testIndexes <- sample(1:nrow(magicGamma), size=0.2*nrow(magicGamma))
testData <- magicGamma[testIndexes,]
trainData <- magicGamma[-testIndexes,]

# Create random bootstrap training samples (with replacement)   ???
trainSamples <- for(i in 1:sampleCount){
  trainData[sample(1:nrow(trainData)
                   , size=0.2*nrow(trainData), replace=TRUE),]
} 

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

#number of bootstrap samples to create
sampleCount <- 8

#-----------------------glm serial-----------------------    
timer <- proc.time()
glmSerial <- foreach(i = 1:sampleCount) %do% {
  glm(V11 ~ .,trainSamples[[i]][,-1], family=binomial())
}
proc.time() - timer 

#-----------------------glm parallel-----------------------
timer <- proc.time()
glmSerial <- foreach(i = 1:sampleCount) %dopar% {
  glm(V11 ~ .,trainSamples[[i]][,-1], family=binomial())
}
proc.time() - timer                

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



#-----------------------ada serial-----------------------
timer <- proc.time()
modelDataAda <- foreach(i = 1:sampleCount) %do% {
  library(ada)
  ada(x = trainSamples[[i]][,c(-1,-ncol(testData))]
      , y = as.numeric(trainSamples[[i]][,ncol(testData)]) - 1)
}
proc.time() - timer

#-----------------------ada-----------------------
timer <- proc.time()
modelDataAda <- foreach(i = 1:sampleCount) %dopar% {
  library(ada)
  ada(x = trainSamples[[i]][,c(-1,-ncol(testData))]
      , y = as.numeric(trainSamples[[i]][,ncol(testData)]) - 1)
}             
proc.time() - timer                

#-------------------------------------------------------------------------------
#-----------------------Practical Example Code Below----------------------------
#-------------------------------------------------------------------------------


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

#-----------------------glm all-----------------------    
timer <- proc.time()
glmAll <- glm(V11 ~ .,trainData[,-1], family=binomial())
proc.time() - timer 

timer <- proc.time()
glmAllTest <- predict(glmAll, testData[,c(-1 ,-ncol(testData))])
proc.time() - timer 

#add predicted class and actual class to test data
glmAllResults <- data.frame(id = testData$id, actualClass = testData$V11)  
glmAllResults$predictedClass <- ifelse(glmAllTest < 0,"g","h")

#calculate  glm all model accuracy
accuracy(glmAllResults$actualClass,glmAllResults$predictedClass)                

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

#-----------------------ada all-----------------------                
timer <- proc.time()
library(ada)
adaAll <- ada(x = trainData[,c(-1,-ncol(trainData))]
              , y = as.numeric(trainData[,ncol(trainData)]) - 1)
proc.time() - timer

timer <- proc.time()
adaAllTest <- predict(adaAll, testData[,c(-1 ,-ncol(testData))], type="probs")
proc.time() - timer 

#add predicted class and actual class to test data
adaAllResults <- data.frame(id = testData$id, actualClass = testData$V11)
adaAllResults$predictedClass <- ifelse(adaAllTest[,1] > adaAllTest[,2],"g","h") 

#calculate ada all model accuracy
accuracy(adaAllResults$actualClass,adaAllResults$predictedClass) 


#-------------------------------------------------------------------------------
#-----------------------Create Different Bootstrap Models in Parallel-----------
#-------------------------------------------------------------------------------

#-----------------------glm-----------------------
timer <- proc.time()
modelDataGlm <- foreach(i = 1:sampleCount) %dopar% {
  glm(V11 ~ .,trainSamples[[i]][,-1], family=binomial())
}                
proc.time() - timer  

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

#-----------------------ada-----------------------
timer <- proc.time()
modelDataAda <- foreach(i = 1:sampleCount) %dopar% {
  library(ada)
  ada(x = trainSamples[[i]][,c(-1,-ncol(testData))]
      , y = as.numeric(trainSamples[[i]][,ncol(testData)]) - 1)
}
proc.time() - timer 

#-------------------------------------------------------------------------------
#-----------------------Predict the Test Data in Parallel Using Each Model------
#-------------------------------------------------------------------------------

#-----------------------glm-----------------------
timer <- proc.time()
predictDataGlm <- foreach(i = 1:sampleCount) %dopar% {
  predict(modelDataGlm[[i]], testData[,c(-1 ,-ncol(testData))]) 
}                
proc.time() - timer

#-----------------------svm-----------------------
timer <- proc.time()
predictDataSvm <- foreach(i = 1:sampleCount) %dopar% {
  library(e1071)
  predict(modelDataSvm[[i]], testData[,c(-1 ,-ncol(testData))]
          , probability=TRUE)
}
proc.time() - timer

#-----------------------ada-----------------------
timer <- proc.time()
predictDataAda <- foreach(i = 1:sampleCount) %dopar% {
  predict(modelDataAda[[i]], testData[,c(-1 ,-ncol(testData))]
          , type="probs")
}                
proc.time() - timer

#-------------------------------------------------------------------------------
#-----------------------Rank Each Model's Bootstap Data-------------------------
#-------------------------------------------------------------------------------
rankPredictData <- function(predictData, getPofG, rankDataObject=NULL,reverseRank=FALSE, addFinalRank=FALSE){
  rankData <- rankDataObject
  #Rank the test data's probability of g for each model
  rankCols <- foreach(i = 1:length(predictData)) %dopar% {
    if(getPofG == "svm"){
      Pg <- attr(predictData[[i]], "probabilities")[,c("g")]
      g  <- ifelse(Pg >= 0.5,1,0)
    } else if (getPofG == "ada"){
      Pg <- predictData[[i]][,1]
      g  <- ifelse(predictData[[i]][,1] >= predictData[[i]][,2],1,0)
    } else {
      Pg <- predictData[[i]]
      g  <- ifelse(Pg <=0,1,0)
    }
    if(reverseRank){ Pg <- -Pg }
    data.frame(gVote=g, rankG=rank(Pg))
  }
  #convert list into one data frame
  colOffset<-ifelse(is.null(rankData),0,ncol(rankData)-1)
  newRankData <- rankCols[[1]]
  colnames(newRankData)[2] <- paste("rank",colOffset + 1,sep='')
  for(i in 2:length(rankCols)){
    newRankData <- data.frame(newRankData, rankCols[[i]]$rankG)
    newRankData$gVote <- newRankData$gVote + rankCols[[i]]$gVote
    colnames(newRankData)[i+1] <- paste("rank",i + colOffset,sep="")
  }
  #Combine any previous rankData, if provided
  if(is.null(rankData)){ 
    rankData <- newRankData
  } else {
    rankData <- data.frame(rankData,newRankData[,-1])
    rankData$gVote <- rankData$gVote + newRankData$gVote 
  }
  if(addFinalRank){
    #create final ranking by summing all ranks and then sort DESC
    rankData$finalRank <- apply(rankData[,-1],1,sum)      
    #add ground truth class column to output
    rankData <- data.frame(id=testData$id,rankData,actualClass=testData[,ncol(testData)])
    #Sort by the rank of class g
    rankData <- rankData[with(rankData, order(-finalRank)), ]
    #add predicted class column to output
    rankData$predictedClass <- ifelse(rankData$gVote >= ((ncol(rankData)-1) / 2),"g","h")
  }
  return(rankData)
}

#-----------------------Create the final ranking tables------------------- 
rankGlm <- rankPredictData(predictDataGlm,"glm",NULL,TRUE,TRUE)
rankSvm <- rankPredictData(predictDataSvm,"svm",NULL,FALSE,TRUE)
rankAda <- rankPredictData(predictDataAda,"ada",NULL,FALSE,TRUE)

#-------------------------------------------------------------------------------
#-----------------------Calculate Accuracy for each model-----------------------
#-------------------------------------------------------------------------------

#-----------------------topN Accuracy Function----------------------------------
rankTopN <- function(rankData) {
  rankAll<-rankData
  topN <- 150
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
  topN <- 175
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
  topN <- 250
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
  topN <- 500
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
  topN <- 1000
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
  topN <- 1500
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
  topN <- 2000
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
  topN <- 2488
  print(accuracy(rankAll[1:topN,]$actualClass,rankAll[1:topN,]$predictedClass))
}

#-----------------------Truth Table and Accuracy - Bootstrap Models-----------------
# glm accuracy
accuracy(rankGlm$actualClass,rankGlm$predictedClass)
# svm accuracy
accuracy(rankSvm$actualClass,rankSvm$predictedClass)
# ada accuracy
accuracy(rankAda$actualClass,rankAda$predictedClass)

#-----------------------TopN Accuracy - Bootstrap Models----------------------------
rankTopN(rankGlm)
rankTopN(rankSvm)
rankTopN(rankAda)

#-----------------------TopN Accuracy - Blended Bootstrap Models----------------------------
rankGlmAll <- rankPredictData(predictDataGlm,"glm",NULL,TRUE,FALSE)
rankGlmSvm <- rankPredictData(predictDataSvm,"svm",rankGlmAll,FALSE,FALSE)
rankAll <- rankPredictData(predictDataAda,"ada",rankGlmSvm,FALSE,TRUE)

rankTopN(rankAll)

#-------------------------------------------------------------------------------
#-----------------------Stacking top 2000 SVM and ADA Predictions---------------
#-------------------------------------------------------------------------------
rankGlm <- rankPredictData(predictDataGlm,"glm",NULL,TRUE,TRUE)
rankSvm <- rankPredictData(predictDataSvm,"svm",NULL,FALSE,TRUE)
rankAda <- rankPredictData(predictDataAda,"ada",NULL,FALSE,TRUE)
#get top 2000 predictions for each model
rankSvm <- rankSvm[1:2000,] 
rankAda <- rankAda[1:1500,]
rankGlm <- rankGlm[1:100,]

#Find Ada records missing from Svm
rankSvmAdaMatches <- which(rankSvm$id %in% rankAda$id)
rankAdaMissing <- rankAda[-rankSvmAdaMatches,]

#combine Svm and missing Ada
rankSvmAda <- rbind(rankSvm,rankAdaMissing)

#Find glm records missing from SvmAda
rankGlmSvmAdaMatches <- which(rankGlm$id %in% rankSvmAda$id)
rankGlmMissing <- rankGlm[-rankGlmSvmAdaMatches,]

#combine Svm and missing Ada
rankGlmSvmAda <- rbind(rankSvmAda,rankGlmMissing)

#check accuracy rankSvmAda
accuracy(rankGlmSvmAda$actualClass,rankGlmSvmAda$predictedClass)
nrow(rankGlmSvmAda)

