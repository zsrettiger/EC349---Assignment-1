# EC349---Assignment-1
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
