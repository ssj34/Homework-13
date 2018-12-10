library(dplyr)
library(caret)
library(nnet)

show_number <- function(m, i, oriented=T)
{
  im <- matrix(mtrain[i,], byrow=T, nrow=28)
  
  if (oriented) {
    im_orient <- matrix(0, nrow=28, ncol=28)
    for (i in 1:28)
      im_orient[i,] <- rev(im[,i])
    
    im <- im_orient
  }
  image(im)
}


# get the training datasets
# I would like to relabel the first column of mtrain, which indicates the actual number.
# If the actual number is 3 then "1" sits in the first coulmn, and "0" if its not.
if (!exists("mtrain")) {
  mtrain <- read.csv("mnist_train.csv", header=F) %>% as.matrix
  for (i in 1:nrow(mtrain)){
    cn <- mtrain[i,1]
    if (cn == "3"){
      mtrain[i,1] <- 1
    } else {
      mtrain[i,1] <- 0
    }
  }
  train_classification <- mtrain[,1]
  mtrain <- mtrain[,-1]/256
  colnames(mtrain) <- 1:784
  rownames(mtrain) <- NULL
}



dim(mtrain) #this shows that I have removed the sample label

#Train classification 

# look at a sample
show_number(mtrain, 8)



#######################################################################################
#Train the neural net to identify the number 3. # To train this neural net I will only use the frist 1000 mnist samples
mtrain <- mtrain[1:1000,]

y <- factor(train_classification[1:1000])
x <- mtrain

prediction_errors <- function(classification, unseen_data, nnet)
{
  y <- factor(classification[1:1000])
  x <- unseen_data
  
  true_y <- y
  pred_y <- predict(nnet, x)
  
  n_samples <- nrow(x)
  error <- sum(true_y != pred_y)/n_samples
  return (error)
}

###################################################
# part i)
tuning_df <- data.frame(size=10, decay=0)

fitControl <- trainControl(## 2-fold CV
  method = "repeatedcv",
  number = 2, 
  repeats = 3)

t_out <- caret::train(x=x, y=y, method="nnet",
                     trControl = fitControl,
                     tuneGrid=tuning_df, maxit = 10000, MaxNWts = 10000) %>% invisible


###################################################
# part ii)
tuning_df <- data.frame(size=4:8, decay=c(.6,.7,.8,.9,1))

fitControl <- trainControl(## 2-fold CV
  method = "repeatedcv",
  number = 2, 
  repeats = 3)

t_out2 <- caret::train(x=x, y=y, method="nnet",
                      trControl = fitControl,
                      tuneGrid=tuning_df, maxit = 10000, MaxNWts = 10000)

###################################################
#Testing the neural network on unseen data 

if (!exists("mtest")) {
  mtest <- read.csv("mnist_test.csv", header=F) %>% as.matrix
  for (i in 1:6000){
    cn <- mtest[i,1]
    if (cn == "3"){
      mtest[i,1] <- 1
    } else {
      mtest[i,1] <- 0
    }
  }
  test_classification <- mtest[,1]
  mtest <- mtest[,-1]/256 
  colnames(mtest) <- 1:784
  rownames(mtest) <- NULL
}
mtest <- mtest[1:1000,]

prediction_errors(test_classification,mtest,t_out)
prediction_errors(test_classification,mtest,t_out2)
