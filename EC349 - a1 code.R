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
library(ipred) 
library(caret)  
library(randomForest) 

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

##Decision Tree, With rpart library
  #Classification Tree
  rpart_treec<-rpart(stars ~ ., data = train, method = 'class')
  rpart.plot(rpart_treec)
  summary(rpart_treec)
  
    #Fit on Test Data
    rpart_predictionc <- predict(rpart_treec, newdata = test, type="class")
  
    #Calculate accuracy
    # Divide the correctly specified number of stars by the number of stars in the test set 
    accuracy_rpartc <- sum(rpart_predictionc == test_y) / nrow(test)
    cat('Accuracy for Classification Tree (rpart):', accuracy_rpartc, '\n')
  
  #Regression Tree
  rpart_treer<-rpart(stars ~ ., data = train, method = 'anova')
  rpart.plot(rpart_treer)
  summary(rpart_treec)
  
  #Fit on Test Data
    rpart_predictionr <- predict(rpart_treer, newdata = test)
    
    rpart_MSEr<- mean((rpart_predictionr - test_y) ^ 2) 
    cat('MSE for Regression Tree (rpart):', rpart_MSEr, '\n')
    
  
    #Calculate accuracy
    # Convert predicted values to rounded stars (assuming stars are integers)
    # Count the number of correctly predicted stars
    # Divide the correctly specified number of stars by the number of stars in the test set 
    accuracy_rpartr <- sum(round(rpart_predictionr) == test_y) / nrow(test)
    cat('Accuracy for Regression Tree (rpart):', accuracy_rpartr, '\n')
    

#Bagging
set.seed(123)     
bag <- bagging(stars ~., data = train, nbagg = 20,   
                 coob = TRUE, control = rpart.control(minsplit = 2, cp = 0.1))
  
  #display fitted bagged model
  bag
  
  #Fit on Test Data
  bag_predictions <- predict(bag, newdata = test)
  
  bag_MSE<- mean((bag_predictions - test_y) ^ 2) 
  cat('MSE for Bagging:', bag_MSE, '\n')
  
  #Calculate accuracy
  # Convert predicted values to rounded stars (assuming stars are integers)
  # Count the number of correctly predicted stars
  # Divide the correctly specified number of stars by the number of stars in the test set 
  accuracy_bag <- sum(round(bag_predictions) == test_y) / nrow(test)
  cat('Accuracy for Bagging:', accuracy_bag, '\n')
  
  
## Random Forest
stars_f <- as.factor(train$stars)
set.seed(12)
model_RF<-randomForest(stars_f ~.,data=train, ntree=20)
model_RF
pred_RF_test = predict(model_RF, test)
RF_err <- mean(model_RF[["err.rate"]])
cat('Error rate for Random Forest:', RF_err, '\n')

#Calculate accuracy
# Count the number of correctly predicted stars
# Divide the correctly specified number of stars by the number of stars in the test set 
accuracy_rf <- sum(pred_RF_test == test_y) / nrow(test)
cat('Accuracy for Random Forest:', accuracy_rf, '\n')
importance(model_RF)


##Linear Regression
lm_stars <- lm(stars ~ average_stars + funny.x + compliment_cool + 
                 compliment_cute + compliment_more + compliment_note +
                 compliment_photos + review_count, data = train) 
summary(lm_stars) 

#Prediction to test data
lm_stars_predict<-predict(lm_stars, newdata = test)

#Empirical MSE in TEST data
lm_stars_test_MSE<-mean((lm_stars_predict-test$stars)^2)
cat('MSE for OLS:', lm_stars_test_MSE, '\n')

#Calculate accuracy
# Convert predicted values to rounded stars (assuming stars are integers)
# Count the number of correctly predicted stars
# Divide the correctly specified number of stars by the number of stars in the test set 
accuracy_lm <- sum(round(lm_stars_predict) == test_y) / nrow(test)
cat('Accuracy for OLS:', accuracy_lm, '\n')

##Ridge with Cross-Validation 
cv.out <- cv.glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 0)
plot(cv.out)
lambda_ridge_cv <- cv.out$lambda.min #cross-validation is the lambda minimising empirical MSE in training data

#Estimate Ridge with lambda chosen by Cross validation
ridge.mod<-glmnet(train_x, train_y, alpha = 0, lambda = lambda_ridge_cv, thresh = 1e-12)

#Fit on Test Data
ridge.pred <- predict(ridge.mod, s = lambda_ridge_cv, newx = as.matrix(test_x))
ridge_MSE<- mean((ridge.pred - test_y) ^ 2) #Outperforms OLS
cat('MSE for RIDGE:', ridge_MSE, '\n')

#Calculate accuracy
# Convert predicted values to rounded stars (assuming stars are integers)
# Count the number of correctly predicted stars
# Divide the correctly specified number of stars by the number of stars in the test set 
accuracy_ridge <- sum(round(ridge.pred) == test_y) / nrow(test)
cat('Accuracy for RIDGE:', accuracy_ridge, '\n')

##LASSO with Cross-Validation 
cv.out <- cv.glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 1, nfolds = 3)
plot(cv.out)
lambda_LASSO_cv <- cv.out$lambda.min #cross-validation is the lambda minimising empirical MSE in training data

#Estimate LASSO with lambda chosen by Cross validation
LASSO.mod<-glmnet(train_x, train_y, alpha = 1, lambda = lambda_LASSO_cv, thresh = 1e-12)
coef(LASSO.mod) 

#Fit on Test Data
LASSO.pred <- predict(LASSO.mod, s = lambda_LASSO_cv, newx = as.matrix(test_x))
LASSO_MSE <- mean((LASSO.pred - test_y) ^ 2) #Note how it outperforms OLS
cat('MSE for LASSO:', LASSO_MSE, '\n')

#Calculate accuracy
# Convert predicted values to rounded stars (assuming stars are integers)
# Count the number of correctly predicted stars
# Divide the correctly specified number of stars by the number of stars in the test set 
accuracy_lasso <- sum(round(LASSO.pred) == test_y) / nrow(test)
cat('Accuracy for LASSO:', accuracy_lasso, '\n')
