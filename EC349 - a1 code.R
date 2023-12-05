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
    correctly_predicted_rpartc <- sum(rpart_predictionc == test_y)
    # Divide the correctly specified number of stars by the number of stars in the test set 
    accuracy_rpartc <- correctly_predicted_rpartc/10000
    sum(accuracy_rpartc)
  
  #Regression Tree
  rpart_treer<-rpart(stars ~ ., data = train, method = 'anova')
  rpart.plot(rpart_treer)
  summary(rpart_treec)
  
  #Fit on Test Data
    rpart_predictionr <- predict(rpart_treer, newdata = test)
    rpart_MSEr<- mean((rpart_predictionr - test_y) ^ 2) #Note how it outperforms OLS
    summary(rpart_MSEr)
  
    #Calculate accuracy
    # Convert predicted values to rounded stars (assuming stars are integers)
    rounded_predictions_rpartr <- round(rpart_predictionr)
    # Count the number of correctly predicted stars
    correctly_predicted_rpartr <- sum(rounded_predictions_rpartr == test_y)
    # Divide the correctly specified number of stars by the number of stars in the test set 
    accuracy_rpartr <- correctly_predicted_rpartr/10000
    sum(accuracy_rpartr)  

#Bagging
set.seed(123)     
bag <- bagging(stars ~., data = train, nbagg = 20,   
                 coob = TRUE, control = rpart.control(minsplit = 2, cp = 0.1))
  
  #display fitted bagged model
  bag
  
  #Fit on Test Data
  bag_predictions <- predict(bag, newdata = test)
  bag_MSE<- mean((bag_predictions - test_y) ^ 2) #Note how it outperforms OLS
  summary(bag_MSE)
  
  #Calculate accuracy
  # Convert predicted values to rounded stars (assuming stars are integers)
  rounded_predictions_bag <- round(bag_predictions)
  # Count the number of correctly predicted stars
  correctly_predicted_bag <- sum(rounded_predictions_bag == test_y)
  # Divide the correctly specified number of stars by the number of stars in the test set 
  accuracy_bag <- correctly_predicted_bag/10000
  sum(accuracy_bag)
  
## Random Forest
stars_f <- as.factor(train$stars)
set.seed(12)
model_RF<-randomForest(stars_f ~.,data=train, ntree=20)
model_RF
pred_RF_test = predict(model_RF, test)
RF_err <- mean(model_RF[["err.rate"]])
summary(RF_err)

# Count the number of correctly predicted stars
correctly_predicted_rf <- sum(pred_RF_test == test_y)
# Divide the correctly specified number of stars by the number of stars in the test set 
accuracy_rf <- correctly_predicted_rf/10000
sum(accuracy_rf)


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
