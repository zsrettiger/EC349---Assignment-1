#Clear
cat("\014") 
rm(list=ls())

#Set Directory 
setwd("C:/Users/zsret/Documents/University of Warwick - Year 4/EC349 - Data Science/EC349 - Assignment 1")

#Load Libraries
library(glmnet)
library(ggplot2)
library(tidyverse)
library(tree)
library(rpart)
library(rpart.plot)

library(adabag) #AdaBoost
library(caret)  #create data partition/training dataset
library(randomForest) #randomforest
library(psych)  #Easy package for PCA

#Load Datasets 
load(file='yelp_review_small.Rda')
load(file='yelp_user_small.Rda')
user_review <- merge(review_data_small, user_data_small,by = "user_id")

#Chech the variables in the data set 
str(user_review)

#Create Test and Training Data sets
set.seed(1)
parts <- createDataPartition(user_review$stars, times = 1,  p = 10000/nrow(user_review), list = F)
#Exclude variables that are characters, for the rest of the variables: stars and average_stars are numeric and the rest are integers
test <- user_review[parts,-c(1,2,3,8,9,10,12,16,17)]
test_x <-test[,-1] 
test_y <-test[,1]
train <- user_review[-parts, -c(1,2,3,8,9,10,12,16,17)]
train_x <-train[,-1]
train_y <-train[,1]
nrow(test)

##Linear Regression
lm_stars <- lm(stars ~ average_stars + funny.x + compliment_cool + 
                  compliment_cute + compliment_more + compliment_note +
                  compliment_photos + review_count, data = train) 
summary(lm_stars) 

  #Prediction to test data
  lm_stars_predict<-predict(lm_stars, newdata = test)

  #Empirical MSE in TEST data
  lm_stars_test_MSE<-mean((lm_stars_predict-test$stars)^2)
  summary(lm_stars_test_MSE) 
  
  #Calculate accuracy
     # Convert predicted values to rounded stars (assuming stars are integers)
       rounded_predictions_lm <- round(lm_stars_predict)
     # Count the number of correctly predicted stars
       correctly_predicted_lm <- sum(rounded_predictions_lm == test_y)
     # Divide the correctly specified number of stars by the number of stars in the test set 
       accuracy_lm <- correctly_predicted_lm/10000
       sum(accuracy_lm)

##Ridge with Cross-Validation 
cv.out <- cv.glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 0)
plot(cv.out)
lambda_ridge_cv <- cv.out$lambda.min #cross-validation is the lambda minimising empirical MSE in training data
  
  #Estimate Ridge with lambda chosen by Cross validation
  ridge.mod<-glmnet(train_x, train_y, alpha = 0, lambda = lambda_ridge_cv, thresh = 1e-12)
  
  #Fit on Test Data
  ridge.pred <- predict(ridge.mod, s = lambda_ridge_cv, newx = as.matrix(test_x))
  ridge_MSE<- mean((ridge.pred - test_y) ^ 2) #Outperforms OLS
  summary(ridge_MSE)
  
     #Calculate accuracy
     # Convert predicted values to rounded stars (assuming stars are integers)
       rounded_predictions_ridge <- round(ridge.pred)
     # Count the number of correctly predicted stars
       correctly_predicted_ridge <- sum(rounded_predictions_ridge == test_y)
     # Divide the correctly specified number of stars by the number of stars in the test set 
       accuracy_ridge <- correctly_predicted_ridge/10000
       sum(accuracy_ridge)
  
  
##LASSO with Cross-Validation 
cv.out <- cv.glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 1, nfolds = 3)
plot(cv.out)
lambda_LASSO_cv <- cv.out$lambda.min #cross-validation is the lambda minimising empirical MSE in training data
  
  #Estimate LASSO with lambda chosen by Cross validation
  LASSO.mod<-glmnet(train_x, train_y, alpha = 1, lambda = lambda_LASSO_cv, thresh = 1e-12)
  coef(LASSO.mod) #note that some parameter estimates are set to 0 --> Model selection!
  
  #Fit on Test Data
  LASSO.pred <- predict(LASSO.mod, s = lambda_LASSO_cv, newx = as.matrix(test_x))
  LASSO_MSE<- mean((LASSO.pred - test_y) ^ 2) #Note how it outperforms OLS
  summary(LASSO_MSE)
  
  #Calculate accuracy
  # Convert predicted values to rounded stars (assuming stars are integers)
  rounded_predictions_lasso <- round(LASSO.pred)
  # Count the number of correctly predicted stars
  correctly_predicted_lasso <- sum(rounded_predictions_lasso == test_y)
  # Divide the correctly specified number of stars by the number of stars in the test set 
  accuracy_lasso <- correctly_predicted_lasso/10000
  sum(accuracy_lasso)
  

##Regression Tree
  #With rpart library
  rpart_tree<-rpart(stars ~ ., data = train)
  rpart.plot(rpart_tree)
  summary(rpart_tree)
  
  #Fit on Test Data
  rpart_predictions <- predict(rpart_tree, newdata = test)
  rpart_MSE<- mean((rpart_predictions - test_y) ^ 2) #Note how it outperforms OLS
  summary(rpart_MSE)
  

#Bagging
  library(ipred)       #for fitting bagged decision trees
  set.seed(1312)       #make this example reproducible
  
  #fit the bagged model
  bag <- bagging(stars ~., data = train, nbagg = 20,   
                 coob = TRUE, control = rpart.control(minsplit = 2, cp = 0.1))
  
  #display fitted bagged model
  bag
  
  bag_predictions <- predict(bag, newdata = test)
  bag_MSE<- mean((bag_predictions - test_y) ^ 2) #Note how it outperforms OLS
  summary(rpart_MSE)
  
  #Calculate accuracy
  # Convert predicted values to rounded stars (assuming stars are integers)
  rounded_predictions_bag <- round(bag_predictions)
  # Count the number of correctly predicted stars
  correctly_predicted_bag <- sum(rounded_predictions_bag == test_y)
  # Divide the correctly specified number of stars by the number of stars in the test set 
  accuracy_bag <- correctly_predicted_bag/10000
  sum(accuracy_bag)
  
## Random Forest
set.seed(1312)
model_RF<-randomForest(as.numeric(stars)~.,data=train, ntree=50)
pred_RF_test = predict(model_RF, test)
mean(model_RF[["err.rate"]])

RF_MSE<- mean((pred_RF_test - test_y) ^ 2) #Note how it outperforms OLS
summary(RF_MSE)

#Calculate accuracy
# Convert predicted values to rounded stars (assuming stars are integers)
rounded_predictions_bag <- round(bag_predictions)
# Count the number of correctly predicted stars
correctly_predicted_bag <- sum(rounded_predictions_bag == test_y)
# Divide the correctly specified number of stars by the number of stars in the test set 
accuracy_bag <- correctly_predicted_bag/10000
sum(accuracy_bag)

  
## PCA ###
library(psych)
pc <- prcomp(train[,-c(9, 10, 12, 1,2,3,4,8,16,17)], center = TRUE, scale. = TRUE) #remove 5th entry which is non-numeric
attributes(pc)
  


##Boosting
  
  # train a model using our training data
  model_adaboost <- boosting(stars ~ average_stars, data=train, boos=TRUE, mfinal=10, control = rpart.control(minsplit = 0))
  summary(model_adaboost)
  
  mfinal=50
  
  + compliment_cool + 
    compliment_cute + compliment_more + compliment_note +
    compliment_photos + review_count
  
  # Load data for Regression (this one is stored on R)
  #use model to make predictions on test data
  pred_ada_test <- predict(model_adaboost, test)
  
  # Returns the prediction values of test data along with the confusion matrix
  pred_ada_test
  
  



#n_text <- gsub(",", "", train$text) 
#text <- as.numeric(n_text)
#length(text)
  
  set.seed(123)
  stars_tree<-tree(stars ~ average_stars + funny.x + compliment_cool + 
                     compliment_cute + compliment_more + compliment_note +
                     compliment_photos + review_count,data = train)
  
  #Fit on Test Data
  tree_predictions <- predict(stars_tree, newdata = test)
  tree_MSE<- mean((tree_predictions - test_y) ^ 2) #Note how it outperforms OLS
  summary(tree_MSE)
  
  set.seed(123)
  stars_tree<-tree(stars ~.,data = train)
  
  #Fit on Test Data
  tree_predictions <- predict(stars_tree, newdata = test)
  tree_MSE<- mean((tree_predictions - test_y) ^ 2) #Note how it outperforms OLS
  summary(tree_MSE)
  
  #Calculate accuracy
  # Convert predicted values to rounded stars (assuming stars are integers)
  rounded_predictions_tree <- round(tree_predictions)
  # Count the number of correctly predicted stars
  correctly_predicted_tree <- sum(rounded_predictions_tree == test_y)
  # Divide the correctly specified number of stars by the number of stars in the test set 
  accuracy_tree <- correctly_predicted_tree/10000
  sum(accuracy_tree)
  
  #partition graph
  partition.tree(stars_tree) 
  points(train[,c("average_stars" , "funny.x" , , "compliment_cool" , 
                  "compliment_cute" , "compliment_more" , "compliment_note" , 
                  "compliment_photos" , "review_count")], cex=.02)
  
  plot(stars_tree)
  text(tree1, pretty = 1)
  title(main = "Classification Tree")
  


  