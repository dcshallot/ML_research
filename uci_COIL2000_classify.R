
# File-Name:       uci_COIL2000_classify.R           
# Date:            2014-04-13                      
# Author:          Chong Ding (chong.ding83@gmail.com)
# Purpose:         二分类问题研究，考虑不同算法，特征选择和非平衡样本问题
##                  （feature selection & imbalanced data trying and testing ）
## 二分类问题y=0,1是实际业务中经常遇到的，如客户流失、欺诈评估、购买等都属于这类问题
## 实际中大量二分问题都会面临非平衡样本，如999个0VS1个1，一般认为再抽样是比较好的办法
## 本研究尝试评价不同再抽样方法结合不同分类器的效果
# Data Used:       http://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)
# Packages Used:   MASS,rpart,e1071,randomForest,pROC,ROSE,caret
# 发现：
## 使用原始数据，按照TP+TN最大判定原则，各种方法中，SVM和LDA以及随机森出其他方法很多高
## 对非平衡样本的处理采用ROSE包，lda由于无法处理单个自变量全一样而放弃使用
## under sampling，所有算法的TP极大提升且达到相同水平，但是TN减小也很严重，FN增大
## over sampling，回归、决策树TP极大提升，TN略好于under，svm和随机森林各项同原始
## both sampling，对所有算法效果跟over效果差不多，但是计算量下降很多
## 对多变量进行特征选择，caret法，除poission回归有些许改进之外其他效果似乎变差了
# 结论：
## SVM和随机森林算法本身对非平衡样本有一定稳健性，而回归则不
## 流失、欺诈等关注POSITIVE的问题，建议比较both方法的回归和under的全部算法
## 特征选择问题尚无好的解决办法


### data import
train <- read.table(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt", header=F)

test <-cbind( 
  read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt", header=F),
  read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt", header=F ))
head(train)

### simple model

pois <- glm(V86 ~ ., family= poisson(), data=train) ##泊松回归
logi <- glm(V86 ~ ., family= binomial(link="logit"), data=train) ##logisitic回归
library(MASS); lda1 <- lda(V86 ~ ., data=train) ##线性判别分析
##决策树
library(rpart)
tree1 <- rpart(V86 ~ ., data=train, method="poisson") # 二分变量用poisson
plot(tree1);text(tree1)
plotcp(tree1)
printcp(tree1) # node的属性，用于修建树
#支持向量机
library(e1071); svm1 <- svm(V86 ~ ., data=train, cost=100, gama=1)
#随机森林
library(randomForest); rforest <- randomForest(V86 ~ ., data=train)
print(rforest);importance(rforest)

## evaluation
p.pois <- predict(pois, newdata=test[,-86], type="response")
p.logi <- predict(logi, newdata=test[,-86], type="response") 
p.lda1 <- predict(lda1, newdata=test[,-86])$class #lda的预测输出不一样
p.tree1 <- predict(tree1, newdata=test[,-86])
p.svm1 <- predict(svm1, test[,-86] )
p.rforest <- predict(rforest, test[,-86])

# confusion matrix
model1 <- rbind( paste( table(test[,86], p.pois > 0.5)) , 
                  paste(table(test[,86], p.logi > 0.5)),
                  paste(table(test[,86], p.lda1)) ,
                 paste(table(test[,86], p.tree1 > 0.5)) ,
                 paste(table(test[,86], p.svm1 > 0.5 )) ,
                paste(table(test[,86], p.rforest> 0.5)) 
                           )
rownames(model1) <- c("pois","logi","lda","tree","svm","forest")
names(model1) <- c("FN","FP","TN","TP") 
model1[4,3] <- 0 # tree has no positive predicts!!!
model1[4,4] <- 0

# ROC plots
library(pROC)
proc <- par(mfrow=c(2,3))
plot.roc(test[,86],p.pois);title("poission")
plot.roc(test[,86],p.logi);title("logistic")
plot.roc(test[,86],p.tree1);title("tree")
plot.roc(test[,86],p.svm1);title("svm")
plot.roc(test[,86],p.rforest);title("randomforest")
par(proc)



### data restucture - balance data

library(ROSE)  
## The package provides functions to deal with binary classiﬁcation
# problems in the presence of imbalanced classes. Synthetic balanced samples are
# generated according to ROSE (Menardi and Torelli, 2013).
table(train$V86)
table(test[,86])
train <- train[order(train$V86),]

##under-sampling
train.under <- ovun.sample(V86 ~., data=train, p=0.5, seed=1, method="under")$data #both, under, over
table(train.under$V86)
summary(train.under)

##modeling and predicting
#train models

models <- function( input) {
  #input <- train.under
  #model build
  pois <- glm(V86 ~ ., family= poisson(), data=input)
  logi <- glm(V86 ~ ., family= binomial(link="logit"), data=input) 
  library(rpart); tree <- rpart(V86 ~ ., data=input, method="poisson")
  library(e1071); svm <- svm(V86 ~ ., data=input, cost=100, gama=1)
  library(randomForest); rforest <- randomForest(V86 ~ ., data=input)
  #predict
  p.pois <- predict(pois, newdata=test[,-86], type="response")
  p.logi <- predict(logi, newdata=test[,-86], type="response") 
  p.tree <- predict(tree, newdata=test[,-86])
  p.svm <- predict(svm, test[,-86] )
  p.rforest <- predict(rforest, test[,-86])
  #results
  output <- data.frame( y= test[,86], 
                        pois = (p.pois > 0.5),
                        logi = (p.logi > 0.5),
                        tree = (p.tree > 0.5),
                        svm = ( p.svm > 0.5),
                        rforest = (p.rforest > 0.5)
                        )
}

under <- models(train.under)
#evaluation
table( under$y, under$pois)
table( under$y, under$logi)
table( under$y, under$tree)
table( under$y, under$svm)
table( under$y, under$rforest)
# very bad, FP much bigger!
# 砍掉了很多0样本，导致模型对0的预测准确率极大降低

## oversampling
train.over <- ovun.sample(V86 ~., data=train, p=0.5, seed=1, method="over")$data #both, under, over
table(train.over$V86)
over <- models(train.over)
table( over$y, over$pois)
table( over$y, over$logi)
table( over$y, over$tree)
table( over$y, over$svm)
table( over$y, over$rforest)

## both
train.both <- ovun.sample(V86 ~., data=train, p=0.5, seed=1, method="both")$data #both, under, over
table(train.both$V86)
both <- models(train.both)
table( both$y, both$pois)
table( both$y, both$logi)
table( both$y, both$tree)
table( both$y, both$svm)
table( both$y,both$rforest)

### feature selection
library(caret) # 加载扩展包，处理自变量集x和因变量y
x <- train[ ,-86]
y <- train[, 86]
# 先删去近似于常量的变量
zerovar <- nearZeroVar(x)
nx <- x[,-zerovar]
# 再删去相关度过高的自变量
descrCorr <- cor(nx, use = "na.or.complete", method = "spearman" )
highCorr <- findCorrelation(descrCorr, 0.90)
nx <- nx[, -highCorr]
# 数据预处理步骤（标准化，缺失值处理）
Process <- preProcess(nx)
nx <- predict(Process, nx)
# 用sbf函数实施过滤方法，这里是用随机森林来评价变量的重要性
data.filter <- sbf(nx, y, 
                   sbfControl = sbfControl(functions=rfSBF,
                                           verbose=F,
                                           method='cv'))
# 根据上面的过滤器筛选变量
train.fs <- nx[data.filter$optVariables]
train.fs <- cbind(train.fs, train[,86]);names(train.fs)[16] <- "V86"
out.fs <- models(train.fs)
table( out.fs$y, out.fs$pois)
table( out.fs$y, out.fs$logi)
table( out.fs$y, out.fs$tree)
table( out.fs$y, out.fs$svm)
table( out.fs$y, out.fs$rforest)

