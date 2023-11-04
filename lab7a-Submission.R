
# STEP 1. Install and Load the Required Packages ----
## stats ----
if (require("stats")) {
  require("stats")
} else {
  install.packages("stats", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## MASS ----
if (require("MASS")) {
  require("MASS")
} else {
  install.packages("MASS", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## glmnet ----
if (require("glmnet")) {
  require("glmnet")
} else {
  install.packages("glmnet", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## kernlab ----
if (require("kernlab")) {
  require("kernlab")
} else {
  install.packages("kernlab", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## rpart ----
if (require("rpart")) {
  require("rpart")
} else {
  install.packages("rpart", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}



library(readr)
cz <- read_csv("data/cubic_zirconia.csv")
View(cz)





# A. Linear Algorithms ----
## 1. Linear Regression ----
### 1.a. Linear Regression using Ordinary Least Squares without caret ----
# The lm() function is in the stats package and creates a linear regression
# model using ordinary least squares (OLS).



# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.8,
                                   list = FALSE)
cz_train <-cz [train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
cz_model_lm <- lm(price ~ ., cz_train)

#### Display the model's details ----
print(cz_model_lm)

#### Make predictions ----
predictions <- predict(cz_model_lm, cz_test[, 1:10])

#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

### 1.b. Linear Regression using Ordinary Least Squares with caret ----
#### Load and split the dataset ----


# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.8,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_lm <- train(price ~ ., data = cz_train,
                                       method = "lm", metric = "RMSE",
                                       preProcess = c("center", "scale"),
                                       trControl = train_control)

#### Display the model's details ----
print(cz_caret_model_lm)

#### Make predictions ----
predictions <- predict(cz_caret_model_lm,
                       cz_test[, 1:10])

#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

## 2. Logistic Regression ----
### 2.a. Logistic Regression without caret ----
# The glm() function is in the stats package and creates a
# generalized linear model for regression or classification.
# It can be configured to perform a logistic regression suitable for binary
# classification problems.

#### Load and split the dataset ----
library(readr)
BCW <- read_csv("data/BCW.csv")
View(BCW)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(BCW$diagnosis,
                                   p = 0.7,
                                   list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

#### Train the model ----
BCW_model_glm <- glm(diagnosis ~ ., data = BCW_train,
                          family = binomial(link = "logit"))

#### Display the model's details ----
print(BCW_model_glm)

#### Make predictions ----
probabilities <- predict(BCW_model_glm, BCW_test[, 1:31],
                         type = "response")
print(probabilities)
predictions <- ifelse(probabilities > 0.5, 1, 0)
print(predictions)

#### Display the model's evaluation metrics ----
table(predictions, BCW_test$diagnosis)

# Read the following article on how to compute various evaluation metrics using
# the confusion matrix:
# https://en.wikipedia.org/wiki/Confusion_matrix

### 2.b. Logistic Regression with caret ----
#### Load and split the dataset ----

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(BCW$diagnosis,
                                   p = 0.7,
                                   list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

# Convert the target variable to a factor
BCW_train$diagnosis <- as.factor(BCW_train$diagnosis)
BCW_test$diagnosis <- as.factor(BCW_test$diagnosis)

# Train the model
set.seed(7)
BCW_caret_model_logistic <- train(
  diagnosis ~ .,
  data = BCW_train,
  method = "glm",  # Use "glm" for logistic regression
  metric = "Accuracy",  # Use "Accuracy" for classification
  preProcess = c("center", "scale"),  # Standardize data
  trControl = trainControl(method = "cv", number = 5)  # 5-fold cross-validation
)

# Display the model's details
print(BCW_caret_model_logistic)

# Make predictions
predictions <- predict(BCW_caret_model_logistic, newdata = BCW_test)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)

# Create a confusion matrix plot
fourfoldplot(confusion_matrix$table, color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 3. Linear Discriminant Analysis ----
### 3.a. Linear Discriminant Analysis without caret ----
# The lda() function is in the MASS package and creates a linear model of a
# multi-class classification problem.

# Load the BCW dataset
library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")
# Define a 70:30 train:test data split of the dataset.

set.seed(7)
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

# Convert the target variable to a factor
BCW_train$diagnosis <- as.factor(BCW_train$diagnosis)
BCW_test$diagnosis <- as.factor(BCW_test$diagnosis)

# Train the LDA model
lda_model <- lda(diagnosis ~ ., data = BCW_train)

# Make predictions
predictions <- predict(lda_model, BCW_test)

# Display the model's evaluation metrics
confusion_matrix <- table(predictions$class, BCW_test$diagnosis)
print(confusion_matrix)

# Create a confusion matrix plot
fourfoldplot(confusion_matrix, color = c("grey", "lightblue"), main = "Confusion Matrix")


# Read the following article on how to compute various evaluation metrics using
# the confusion matrix:
# https://en.wikipedia.org/wiki/Confusion_matrix

### 3.b.  Linear Discriminant Analysis with caret ----
#### Load and split the dataset ----
# Load the BCW dataset
library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")


# Define a 70:30 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

# Convert the target variable to a factor
BCW_train$diagnosis <- as.factor(BCW_train$diagnosis)
BCW_test$diagnosis <- as.factor(BCW_test$diagnosis)

# Train the LDA model with Leave One Out Cross Validation
train_control <- trainControl(method = "LOOCV")
lda_model <- train(
  diagnosis ~ .,
  data = BCW_train,
  method = "lda",
  metric = "Accuracy",
  preProcess = c("center", "scale"),
  trControl = train_control
)

# Display the model's details
print(lda_model)

# Make predictions
predictions <- predict(lda_model, newdata = BCW_test)

# Display the model's evaluation metrics (confusion matrix)
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)


## 4. Regularized Linear Regression ----
# The glmnet() function is in the glmnet package and can be used for
# both classification and regression problems.
# It can also be configured to perform three important types of regularization:
##    1. lasso,
##    2. ridge and
##    3. elastic net
# by configuring the alpha parameter to 1, 0 or in [0,1] respectively.

### 4.a. Regularized Linear Regression Classification Problem without CARET ----
#### Load the dataset ----

library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")

# Define a 70:30 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

# Convert the target variable to a factor
BCW_train$diagnosis <- as.factor(BCW_train$diagnosis)
BCW_test$diagnosis <- as.factor(BCW_test$diagnosis)

# Train the Regularized Logistic Regression model with Leave One Out Cross Validation
train_control <- trainControl(method = "LOOCV")
lasso_model <- train(
  diagnosis ~ .,
  data = BCW_train,
  method = "glmnet",
  metric = "Accuracy",
  preProcess = c("center", "scale"),
  trControl = train_control
)

# Make predictions
predictions <- predict(lasso_model, newdata = BCW_test)

# Extract the predicted class labels
predicted_labels <- as.factor(predictions)

# Display the model's evaluation metrics (confusion matrix)
confusion_matrix <- confusionMatrix(predicted_labels, BCW_test$diagnosis)
print(confusion_matrix)



### 4.c. Regularized Linear Regression Classification Problem with CARET ----
#### Load and split the dataset ----
library(readr)
cz <- read_csv("data/cubic_zirconia.csv")
# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.7,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_glmnet <-
  train(price ~ .,
        data = cz_train, method = "glmnet",
        metric = "RMSE", preProcess = c("center", "scale"),
        trControl = train_control)

#### Display the model's details ----
print(cz_caret_model_glmnet)

#### Make predictions ----
predictions <- predict(cz_caret_model_glmnet, cz_test[, 1:10])

#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

### 4.d. Regularized Linear Regression Regression Problem with CARET ----
#### Load and split the dataset ----
library(readr)
cz <- read_csv("data/cubic_zirconia.csv")

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.7,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_glmnet <-
  train(price ~ .,
        data = cz_train, method = "glmnet",
        metric = "RMSE", preProcess = c("center", "scale"),
        trControl = train_control)

#### Display the model's details ----
print(cz_caret_model_glmnet)

#### Make predictions ----
predictions <- predict(cz_caret_model_glmnet, cz_test[, 1:10])

#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

# B. Non-Linear Algorithms ----
## 1.  Classification and Regression Trees ----
### 1.a. Decision tree for a classification problem without caret ----
#### Load and split the dataset ----
# B. Non-Linear Algorithms ----
## 1. Classification and Regression Trees ----
### 1.a. Decision tree for a classification problem without caret ----

# Load necessary libraries
library(readr)
library(caret)
library(rpart)

# Load and split the dataset
cz <- read_csv("data/cubic_zirconia.csv")

# Define a 70:30 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(cz$cut, p = 0.7, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the model
cz_model_rpart <- rpart(cut ~ ., data = cz_train)

# Display the model's details
print(cz_model_rpart)

# Make predictions
predictions <- predict(cz_model_rpart, cz_test, type = "class")

# Check unique levels in both datasets
unique_levels_train <- levels(cz_train$cut)
unique_levels_test <- levels(cz_test$cut)

# Identify any extra levels in the test dataset and remove them
extra_levels <- setdiff(unique_levels_test, unique_levels_train)
if (length(extra_levels) > 0) {
  # Remove extra levels from cz_test$cut
  cz_test$cut <- factor(cz_test$cut, levels = unique_levels_train)
}

# Now, calculate the confusion matrix and classification metrics
#confusion_matrix <- confusionMatrix(predictions, cz_test$cut)
#print(confusion_matrix)




### 1.b. Decision tree for a regression problem without CARET ----
#### Load and split the dataset ----
# Load the cubic_zirconia dataset
cz <- read_csv("data/cubic_zirconia.csv")


# Define an 80:20 train:test data split of the dataset.
set.seed(7)  # For reproducibility
train_index <- createDataPartition(cz$cut, p = 0.8, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the model for a classification problem (e.g., predicting "cut")
cz_model_rpart <- rpart(cut ~ ., data = cz_train, method = "class")

# Display the model's details
print(cz_model_rpart)

# Make predictions
predictions <- predict(cz_model_rpart, newdata = cz_test, type = "class")

# Display the model's evaluation metrics (classification report)
confusion_matrix <- table(predictions, cz_test$cut)
print(confusion_matrix)

# Calculate other classification metrics (e.g., accuracy)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy =", sprintf(accuracy, fmt = "%#.4f")))


### 1.c. Decision tree for a classification problem with caret ----
# Load the necessary libraries
library(caret)
library(e1071)

# Load the BCW dataset (adjust the file path accordingly)
BCW <- read.csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")

# Make sure 'diagnosis' is a factor variable
BCW$diagnosis <- as.factor(BCW$diagnosis)

# Define a 70:30 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

# Create a classification tree model using the 'e1071' package
BCW_caret_model_tree <- train(
  diagnosis ~ ., 
  data = BCW_train, 
  method = "rpart",
  metric = "Accuracy",
  trControl = trainControl(method = "cv", number = 5)
)

# Display the model's details
print(BCW_caret_model_tree)

# Make predictions
predictions <- predict(BCW_caret_model_tree, newdata = BCW_test)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)





# Load the necessary libraries
library(caret)
library(e1071)

# Load the BCW dataset (adjust the file path accordingly)
BCW <- read.csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")

# Make sure 'diagnosis' is a factor variable
BCW$diagnosis <- as.factor(BCW$diagnosis)

# Define a 70:30 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

# Create a classification tree model using the 'e1071' package
BCW_caret_model_tree <- train(
  diagnosis ~ ., 
  data = BCW_train, 
  method = "rpart",
  metric = "Accuracy",
  trControl = trainControl(method = "cv", number = 5)
)

# Display the model's details
print(BCW_caret_model_tree)

# Make predictions
predictions <- predict(BCW_caret_model_tree, newdata = BCW_test)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)

# Calculate and display Mean Absolute Error (MAE)
absolute_errors <- abs(as.numeric(predictions) - as.numeric(BCW_test$diagnosis))
mae <- mean(absolute_errors)
print(paste("Mean Absolute Error (MAE) =", sprintf(mae, fmt = "%#.4f")))

## 2.  Naïve Bayes ----
### 2.a. Naïve Bayes Classifier for a Classification Problem without CARET ----
# We use the naiveBayes function inside the e1071 package
#### Load and split the dataset ----
# Load the BCW dataset (adjust the file path accordingly)

# Load the necessary libraries
library(e1071)
library(caret)

BCW <- read.csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")

# Ensure 'diagnosis' is a factor variable
BCW$diagnosis <- as.factor(BCW$diagnosis)

# Define a 70:30 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, -1]  # Exclude 'id' column
BCW_test <- BCW[-train_index, -1]  # Exclude 'id' column

# Train the Naïve Bayes model
BCW_model_glm <- naiveBayes(diagnosis ~ ., data = BCW_train)

# Make predictions
predictions <- predict(BCW_model_glm, BCW_test)

# Display the model's evaluation metrics using the confusion matrix
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)




### 2.b. Naïve Bayes Classifier for a Classification Problem with CARET ----
#### Load and split the dataset ----
# Load the necessary libraries
# Define a 70:30 train:test data split of the dataset.
# Load and split the dataset
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")

# Convert 'cut' to a factor
cz$cut <- as.factor(cz$cut)

# Define a 70:30 train:test data split of the dataset
train_index <- createDataPartition(cz$cut, p = 0.7, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the Naïve Bayes model
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_nb <- train(cut ~ ., data = cz_train, method = "nb", metric = "Accuracy", trControl = train_control)

# Make predictions
predictions <- predict(cz_caret_model_nb, cz_test)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, cz_test$cut)
print(confusion_matrix)



## 3.  k-Nearest Neighbours ----
# The knn3() function is in the caret package and does not create a model.
# Instead it makes predictions from the training dataset directly.
# It can be used for classification or regression.

### 3.a. kNN for a classification problem without CARET's train function ----
#### Load and split the dataset ----
### 3.a. kNN for a classification problem without CARET's train function ----
#### Load and split the dataset ----
# Load and split the dataset
BCW <- read.csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

# Train the kNN model, excluding the 'id' column
BCW_model_knn <- knn3(diagnosis ~ ., data = BCW_train[, -1], k = 3)

# Display the model's details
print(BCW_model_knn)

BCW_test <- BCW_test[, -which(names(BCW_test) == "id")]

# Make predictions, excluding the 'id' column
predictions <- predict(BCW_model_knn, BCW_test, type = "class")




            
            

### 3.c. kNN for a classification problem with CARET's train function ----
#### Load and split the dataset ----
# Load necessary libraries
library(caret)
library(class)

# Load the dataset
BCW <- read.csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")

# Convert "diagnosis" to a factor with two levels
BCW$diagnosis <- as.factor(BCW$diagnosis)

# Define the percentage for the training set
train_percentage <- 0.7

# Create a random index for splitting
set.seed(123)  # For reproducibility
index <- sample(1:nrow(BCW), nrow(BCW) * train_percentage)

# Split the dataset into training and testing sets
BCW_train <- BCW[index, ]
BCW_test <- BCW[-index, ]

# Train the kNN classification model
k <- 3  # You can choose your desired number of neighbors
BCW_model_knn <- knn(train = BCW_train[, -1], test = BCW_test[, -1], cl = BCW_train$diagnosis, k = k)

# Display the model's details
print(BCW_model_knn)

# Make predictions
predictions <- as.factor(BCW_model_knn)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)

# Plot the confusion matrix
fourfoldplot(as.table(confusion_matrix$table), color = c("grey", "lightblue"), main = "Confusion Matrix")



### 3.d. kNN for a regression problem with CARET's train function ----
#### Load and split the dataset ----
# Load and split the dataset
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")

# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.8,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the model
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_knn <- train(price ~ ., data = cz,
                            method = "knn", metric = "RMSE",
                            preProcess = c("center", "scale"),
                            trControl = train_control)

# Display the model's details
print(cz_caret_model_knn)

# Make predictions
# Make predictions
predictions <- predict(cz_caret_model_knn, cz_test)

# The rest of your code for evaluation metrics...


# Display the model's evaluation metrics
# (RMSE, SSR, SST, R Squared, MAE)
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))


## 4.  Support Vector Machine ----
### 4.a. SVM Classifier for a classification problem without CARET ----
#### Load and split the dataset ----
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.7,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the model
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_svm_radial <- train(price ~ ., data = cz_train, method = "svmRadial",
                                   metric = "RMSE", trControl = train_control)

# Display the model's details
print(cz_caret_model_svm_radial)

# Make predictions using the correct model
predictions <- predict(cz_caret_model_svm_radial, cz_test)

# Display the model's evaluation metrics
# (RMSE, SSR, SST, R Squared, MAE)
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

# Calculate SSR
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

# Calculate SST
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

# Calculate R-squared
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

# Calculate MAE
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))


### 4.b. SVM Classifier for a regression problem without CARET ----
#### Load and split the dataset ----

cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")

# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.8,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
cz_model_svm <- ksvm(price ~ ., cz_train, kernel = "rbfdot")

#### Display the model's details ----
print(cz_model_svm)

#### Make predictions ----
predictions <- predict(cz_model_svm, cz_test)

#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

### 4.c. SVM Classifier for a classification problem with CARET ----
# The SVM with Radial Basis kernel implementation can be used with caret for
# classification as follows:
#### Load and split the dataset ----


# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.7,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the model
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_svm_radial <- train(price ~ ., data = cz_train, method = "svmRadial",
                                   metric = "Accuracy", trControl = train_control)

# Display the model's details
print(cz_caret_model_svm_radial)

# Make predictions using the correct model
predictions <- predict(cz_caret_model_svm_radial, cz_test)

# Display the model's evaluation metrics
table(predictions, cz_test$price)
confusion_matrix <- caret::confusionMatrix(predictions, cz_test$price)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


### 4.d. SVM classifier for a regression problem with CARET ----
# The SVM with radial basis kernel implementation can be used with caret for
# regression as follows:
#### Load and split the dataset ----
library(readr)
cz <- read_csv("data/cubic_zirconia.csv")
View(cz)

# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(cz$price,
                                   p = 0.8,
                                   list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_svm_radial <-
  train(price ~ ., data = cz_train,
        method = "svmRadial", metric = "RMSE",
        trControl = train_control)

#### Display the model's details ----
print(cz_caret_model_svm_radial)

#### Make predictions ----
predictions <- predict(cz_caret_model_svm_radial,
                       cz_test[, 1:13])

#### Display the model's evaluation metrics ----
##### RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))

##### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))

##### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))

##### R Squared ----
# We then use SSR and SST to compute the value of R squared.
# The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))

##### MAE ----
# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))

# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint

# References ----

## Kuhn, M., Wing, J., Weston, S., Williams, A., Keefer, C., Engelhardt, A., Cooper, T., Mayer, Z., Kenkel, B., R Core Team, Benesty, M., Lescarbeau, R., Ziem, A., Scrucca, L., Tang, Y., Candan, C., & Hunt, T. (2023). caret: Classification and Regression Training (6.0-94) [Computer software]. https://cran.r-project.org/package=caret # nolint ----

## Leisch, F., & Dimitriadou, E. (2023). mlbench: Machine Learning Benchmark Problems (2.1-3.1) [Computer software]. https://cran.r-project.org/web/packages/mlbench/index.html # nolint ----

## National Institute of price and Digestive and Kidney Diseases. (1999). Pima Indians price Dataset [Dataset]. UCI Machine Learning Repository. https://www.kaggle.com/datasets/uciml/pima-indians-price-database # nolint ----

## Robin, X., Turck, N., Hainard, A., Tiberti, N., Lisacek, F., Sanchez, J.-C., Müller, M., Siegert, S., Doering, M., & Billings, Z. (2023). pROC: Display and Analyze ROC Curves (1.18.4) [Computer software]. https://cran.r-project.org/web/packages/pROC/index.html # nolint ----

## Wickham, H., François, R., Henry, L., Müller, K., Vaughan, D., Software, P., & PBC. (2023). dplyr: A Grammar of Data Manipulation (1.1.3) [Computer software]. https://cran.r-project.org/package=dplyr # nolint ----

## Wickham, H., Chang, W., Henry, L., Pedersen, T. L., Takahashi, K., Wilke, C., Woo, K., Yutani, H., Dunnington, D., Posit, & PBC. (2023). ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics (3.4.3) [Computer software]. https://cran.r-project.org/package=ggplot2 # nolint ----









