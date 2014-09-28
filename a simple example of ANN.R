
irisdata <- read.csv("http://www.heatonresearch.com/dload/data/iris.csv",head=TRUE,sep=",")
irisTrainData = sample(1:150,100)
irisValData = setdiff(1:150,irisTrainData)

library(nnet)
ideal <- class.ind(irisdata$species)
irisANN = nnet(irisdata[irisTrainData,-5], ideal[irisTrainData,], size=10, softmax=TRUE)
predict(irisANN, irisdata[irisValData,-5], type="class")
