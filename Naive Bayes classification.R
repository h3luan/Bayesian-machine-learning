

# spamdata <- read.csv("SMSSPAM.csv")
spamdata <- read.table("SMSSpamCollection.txt",sep="\t", stringsAsFactors = default.stringsAsFactors(), quote = "")

# Preprare training and testing data --------------------------------------
samp <- sample.int(nrow(spamdata),as.integer(nrow(spamdata)*0.2),replace = F)
spamTest <- spamdata[samp,]
spamTrain <- spamdata[-samp,]

ytrain <- as.factor(spamTrain[,1])
xtrain <- as.factor(spamTrain[,2])
ytest <- as.factor(spamTest[,1])
xtest <- as.factor(spamTest[,2])

# Processing text file ----------------------------------------------------

library(tm)
library(SnowballC)
xtrain <- VCorpus(VectorSource(xtrain))
#Remove extra white space
xtrain <- tm_map(xtrain,stripWhitespace)
xtrain <- tm_map(xtrain, removePunctuation)
xtrain <- tm_map(xtrain, removeNumbers)
xtrain <- tm_map(xtrain, content_transformer(tolower))
#removing stop words
xtrain <- tm_map(xtrain, removeWords, stopwords("english"))
xtrain <- tm_map(xtrain,stemDocument)
xtrain <- as.data.frame.matrix(DocumentTermMatrix(xtrain))

xtest <- VCorpus(VectorSource(xtest))
#Remove extra white space
xtest <- tm_map(xtest,stripWhitespace)
xtest <- tm_map(xtest, removePunctuation)
xtest <- tm_map(xtest, removeNumbers)
xtest <- tm_map(xtest, content_transformer(tolower))
#removing stop words
xtest <- tm_map(xtest, removeWords, stopwords("english"))
xtest <- tm_map(xtest,stemDocument)
xtest <- as.data.frame.matrix(DocumentTermMatrix(xtest))


# Fit the Naive Bayes model -----------------------------------------------

library(naivebayes)
library(e1071)
library(pROC)
#Training the Naive Bayes Model
nbmodel <- naive_bayes(xtrain, ytrain, laplace = 3)
#Prediction using trained model
ypred.nb <- predict(nbmodel, xtest, type = "class", threshold = 0.075)
fconvert <- function(x){
  if(x=="spam"){
    y<-1
  }
  else{
    y<-0
  }
  y
}

ytest1 <- sapply(ytest,fconvert,simplify = "array")
ypred1 <- sapply(ypred.nb, fconvert, simplify = "array")

##Receiver operating characteristic curve (ROC): This is used to find the best threshold (operating point of the classifier) for deciding whether a predicted output (usually a score or probability) belongs to class 1 or -1. 

##Sensitivity: % of positives in the test dataset that have been correctly predicted
##Specificity: % of negatives in the test dataset that have been correctly predicted
roc(ytest1,ypred1,plot=T)


table(ytest, ypred.nb)  ##Confusion matrix


tab <- nbmodel$tables

#Posterior mean of a word assigning as ham
fham <- function(x){
  y <- x[1,1]
  y
}

hamvec <- sapply(tab, fham, simplify = "array")
hamvec <- sort(hamvec, decreasing = T)

#Posterior mean of a word assigning as spam
fspam <- function(x){
  y <- x[1,2]
  y
}

spamvec <- sapply(tab, fspam, simplify = "array")
spamvec <- sort(spamvec, decreasing = T)

pp <- cbind(hamvec, spamvec)
print.table(pp)
