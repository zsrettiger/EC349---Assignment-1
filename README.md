<div align="center">
## Prediction of user reviews on Yelp
*University ID: u2080302*

```{r, include=FALSE}
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

library(adabag) #AdaBoost
library(caret)  #create data partition/training dataset
library(randomForest) #randomforest
library(psych)  #Easy package for PCA

#Load Datasets 
load(file='yelp_review_small.Rda')
load(file='yelp_user_small.Rda')
user_review <- merge(review_data_small, user_data_small,by = "user_id")
```

<div align="left">
### Questions
```{r Question 1, echo=FALSE}
#Create Test and Training Data sets
set.seed(1) 
parts <- createDataPartition(user_review$stars, times = 1,  p = 10000/nrow(user_review), list = F)
test <- user_review[parts,-c(1,2,3,8,9,10,12,16,17)] #Exclude non-numeric variables
test_x <-test[,-1] 
test_y <-test[,1]
train <- user_review[-parts, -c(1,2,3,8,9,10,12,16,17)]
train_x <-train[,-1]
train_y <-train[,1]
```



Question 3.: You will write up an analysis of your chosen method and of your results, in 1250 words.


  In this analysis, I use 20 numeric and integer variables from data sets about user characteristics and reviews merged together, containing 289878 observations to predict the number of stars given by user i to business j.  

  When predicting the amount of stars given by user i, both the outcome of interest Y, that is, the number of stars given, and the characteristics X of users can be observed, thus this is a case of supervised learning. Additionally, in the case of this prediction, the outcome variable is categorical, taking on the values of 1,2,3,4 and 5, such that it requires the use of classification methods. Consequentially, the use of categorical decision trees could lead to precise predictions. Decision trees are non-parametric supervised-learning methods, that split the feature space of X characteristics into regions based on informativeness, such that the variable best classifying the data is split first (IBM, 2022). 
  
  In this case, as shown on the decision tree below, the parent node is a five-star review, while the child notes are one- and five-star reviews. Two-, three- and four-star reviews remain unused in the decision tree, suggesting that they did not contribute to the change in entropy in any of the splits that could lead to a gain in prediction.  According to the tree on the training data, average stars given by users is the most important variable when predicting the stars a user will give, with review count, a cool and a useful rating also being significant variables in the prediction. Receiving a review from a user giving less than three stars on average results in a prediction of one star.     

<div align="center">
```{r Question 3: Decision Tree, echo=FALSE }
##Decision Tree, With rpart library
  #Classification Tree
  rpart_treec<-rpart(stars ~ ., data = train, method = 'class')
  rpart.plot(rpart_treec)
```
   
   
  <div align="left">
  To see how accurate these results are, I create an accuracy measure by comparing the number of stars correctly specified by the model in the test set with the observed stars. I use this measure because mean squared error cannot be used as the outcome is categorical. The constructed accuracy measure suggests that the categorical decision tree predicts the stars given by users with a 51.83 percent accuracy in the test set. The prediction of the regression tree has an accuracy of 35.5 percent in the test set when rounding up to the number of stars to integers, showcasing that the categorical tree is a better fit for this data. 
  
  Decision trees have the advantage of being flexible and easy to interpret. However, the prediction results have a high variance. To account for this, I re-estimate the predictions using the method of bagging regression trees, which decrease the variance by averaging over observations. The bagging model has an out-of-bag estimate of root mean squared error of 1.3092 in the training set. The accuracy of the bagging model is 22.16 percent in the test set; thus, it does not outperform the decision tree on this data set. However, it is important to note that the accuracy of the prediction is measured by rounding up the number of stars given to integers, such that this result is not completely accurate. 
  
  I also estimate the predictions using the Random Forest model. Random Forests combine random trees and bagging, such that they remove correlation in the predictions, which can occur in decision trees because the predictions were generated from the same original data set, while also reducing the variance in the prediction in comparison to decision trees. The Random Forests model has an out-of-bag estimate of  error rate of 46 percent on the training data, and predicts the stars given by users with an accuracy of 99.96 percent, and an error rate of 0.03024 on the test set, outperforming the categorical decision tree. 
  
  Following these results, the most accurate method to predict the number of stars given to business j by user i based on the characteristics of the users and their reviews is the Random Forest model. 




##References:
IBM (2022) 'What is a Decision Tree' [online] www.ibm.com. Available at: https://www.ibm.com/topics/decision-trees.
