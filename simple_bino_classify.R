# 那些年我们用过的二分类算法——小白级科普

# 目的
# 简单来说就是手把手的教给你怎么解决Y=0/1这类模型的训练和评测

# 数据
# http://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)
# 烂大街的数据集，大概是保险公司的客户信息，反正Y要么是0要么是1，X有85个之多
# 这个数据已经把用于训练模型的和用于评价模型性能的数据分开了

# 算法
# 泊松回归、Logistic回归、SVM和随机森林
# Packages Used:   e1071,randomForest


### data import
train <- read.table(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt", header=F)

test <-cbind( 
  read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt", header=F),
  read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt", header=F ))
names(test)[86] <- "V86"
head(train)

## models

pois <- glm(V86 ~ ., family= poisson(), data=train)  ##泊松回归
logi <- glm(V86 ~ ., family= binomial(link="logit"), data=train) ##logisitic回归
# 这两个货有什么区别，我不是太清楚，不过似乎Logistic用的多一些

#支持向量机
library(e1071)
svm1 <- svm(V86 ~ ., data=train )

#随机森林
library(randomForest)
rforest <- randomForest(V86 ~ ., data=train)

## evaluation
# 评价模型，主要是混淆矩阵和衍生的查全率、查准率，这里以Y=1为评价目标
rslt <- function( model) { # model test
  z <- ( predict( model, test, type="response") > 0.5 )
  zz<- table(z, test$V86 , dnn =c("pred","act")) # 混淆矩阵
  out <- list( c.m = zz,  
               accuracy = 1-( sum(zz) - sum(diag(zz)))/sum(zz) , # 查准率
               cover = sum(zz[1,2]/sum(zz[,2])) # Y=0查全率
  )
  return(out)
}

rslt( pois)
rslt( logi)
rslt( svm1 )
rslt( rforest)

# 结论：论精度SVM最好，论速度logistic性价比最高。
# 不想玩深的话，把数据处理好，直接套就可以用了。
# 想玩深一点的话有以下议题：
# 1.step的logistic，稳健的Logistic，样本非平衡性处理后的Logistic
# 2.SVM的核函数各种调整，参数的各种调整
# 3.随机森林的各种调整
# 4.决策树、贝叶斯分类器以及其他



