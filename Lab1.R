install.packages('ggplot2', dep = TRUE) #Install the ggplot2 library
install.packages('class', dep = TRUE)
install.packages("gmodels")
library(ggplot2) #Import ggplot2 library
library(class)
library(gmodels)
rm(list = ls()) #Clean all the old variables
set.seed(as.numeric(Sys.time())) #Initialization of seed with timestamp to get real random every execution

#LOADING DATASET
irisData = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=FALSE,sep = ",") #Loading the dataset in the variable irisData
names(irisData) <- c("SL", "SW" , "PL", "PW", "C" ) #Renaming the colums: SL = sepal length in cm, SW = sepal width in cm, PL = petal length in cm, PW = petal width in cm, C = class

#CHECKING LOADED DATA
irisData
ggplot(irisData, aes(SL, SW, color = C)) + geom_point() + labs(color='Class',x ='Sepal length in cm', y ='Sepal width in cm')
ggplot(irisData, aes(PL, PW, color = C)) + geom_point() + labs(color='Class',x ='Petal length in cm', y ='Petal width in cm')

#KMEAN sepal
set.seed(as.numeric(Sys.time()))
irisClusterSepal <- kmeans(irisData[,1:2], length(unique(irisData[,5])))
irisClusterSepal
table(irisClusterSepal$cluster, irisData$C)
irisClusterSepal$cluster <- as.factor(irisClusterSepal$cluster)
ggplot(irisData, aes(SL, SW, color = irisClusterSepal$cluster,shape=C)) + geom_point() +labs(color='Cluster',shape='Class',x ='Sepal length in cm', y ='Sepal width in cm')

#KMEAN petal
set.seed(as.numeric(Sys.time()))
irisClusterPetal <- kmeans(irisData[,3:4], length(unique(irisData[,5])))
irisClusterPetal
table(irisClusterPetal$cluster, irisData$C)
irisClusterPetal$cluster <- as.factor(irisClusterPetal$cluster)
ggplot(irisData, aes(PL, PW, color = irisClusterPetal$cluster,shape=C)) + geom_point() +labs(color='Cluster',shape='Class',x ='Petal length in cm', y ='Petal width in cm')

#KMEAN petal and sepal
set.seed(as.numeric(Sys.time()))
irisCluster <- kmeans(irisData[,1:4], length(unique(irisData[,5])))
irisCluster
table(irisCluster$cluster, irisData$C)
irisCluster$cluster <- as.factor(irisCluster$cluster)
ggplot(irisData, aes(PL, PW, color = irisCluster$cluster,shape=C)) + geom_point() +labs(color='Cluster', shape='Class',x ='Petal length in cm', y ='Petal width in cm')

# kNN
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
iris_normalized <- as.data.frame(lapply(irisData[1:4], normalize))
summary(iris_normalized$sepal_length) # check if normalized

iris_train <- rbind(iris_normalized[1:25,], iris_normalized[51:75,],  iris_normalized[101:125,])
iris_train_labels <- c(irisData[1:25,5], irisData[51:75,5],  irisData[101:125,5])

iris_test <- rbind(iris_normalized[26:50,], iris_normalized[76:100,],  iris_normalized[126:150,])
iris_test_labels <- c(irisData[26:50,5], irisData[76:100,5],  irisData[126:150,5])

iris_test_pred <- knn(train = iris_train, test = iris_test,cl = iris_train_labels, k=sqrt(nrow(irisData)))
CrossTable(x = iris_test_labels, y = iris_test_pred, prop.chisq = FALSE)

accuracy = function(actual, predicted) { mean(actual == predicted) }
accuracy(actual = iris_test_labels, iris_test_pred)

k_max <- nrow(iris_train)/3

results <- vector("list", k_max)

for (k in 1:k_max){
  iris_test_pred <- knn(train = iris_train, test = iris_test,cl = iris_train_labels, k=k)
  a <- accuracy(actual = iris_test_labels, iris_test_pred)
  results[k] <- a
}

df <- data.frame(accuracy=matrix(unlist(results), nrow=k_max, byrow=T))
df$k <- as.numeric(row.names(df))

ggplot(data=df, aes(x=k, y=accuracy, group=1)) +
  geom_line(color = "red")+
  geom_point()


