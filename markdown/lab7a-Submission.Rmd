---
title: "Business Intelligence Lab Submission Markdown"
author: "Team Marafiki"
date: "31/10/2023"
output:
  github_document: 
    toc: yes
    toc_depth: 4
    fig_width: 6
    fig_height: 4
    df_print: default
editor_options:
  chunk_output_type: console
---

# Student Details

+---------------------------------------------------+---------------------------------------------+
| **Student ID Numbers and Names of Group Members** | 1.  136446 - C - Mirav Bhojani              |
|                                                   |                                             |
|                                                   | 2.  136788 - C - Derrick Nyaga              |
|                                                   |                                             |
|                                                   | 3.  136709 - C - Jane Mugo                  |
|                                                   |                                             |
|                                                   | 4.  136895 - C - Wesley Wanyama             |
|                                                   |                                             |
|                                                   | 5.  135399 - C - Sheilla Kavinya            |
+---------------------------------------------------+---------------------------------------------+
| **GitHub Classroom Group Name**                   | *Team Marafiki*                             |
+---------------------------------------------------+---------------------------------------------+
| **Course Code**                                   | BBT4206                                     |
+---------------------------------------------------+---------------------------------------------+
| **Course Name**                                   | Business Intelligence II                    |
+---------------------------------------------------+---------------------------------------------+
| **Program**                                       | Bachelor of Business Information Technology |
+---------------------------------------------------+---------------------------------------------+
| **Semester Duration**                             | 21^st^ August 2023 to 28^th^ November 2023  |
+---------------------------------------------------+---------------------------------------------+

# Setup Chunk

We start by installing all the required packages

```{r load_packages}
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
```

------------------------------------------------------------------------

**Note:** the following "*KnitR*" options have been set as the defaults in this markdown:\
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy.opts = list(width.cutoff = 80), tidy = TRUE)`.

More KnitR options are documented here <https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and here <https://yihui.org/knitr/options/>.

```{r setup, echo=TRUE, message=FALSE, warning=FALSE}
knitr::opts_chunk$set(
	eval = TRUE,
	echo = TRUE,
	warning = FALSE,
	collapse = FALSE,
	tidy = TRUE
)
```

------------------------------------------------------------------------

**Note:** the following "*R Markdown*" options have been set as the defaults in this markdown:

> output:\
> \
> github_document:\
> toc: yes\
> toc_depth: 4\
> fig_width: 6\
> fig_height: 4\
> df_print: default\
> \
> editor_options:\
> chunk_output_type: console

# Load the dataset

Load the BreastCancer dataset

```{r Load the dataset}
library(readr)
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")
#View(cz)
```

# Linear Regression

```{r Linear Regression}
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

```

# Linear Regression using Ordinary Least Squares with caret

```{r Linear Regression using Ordinary Least Squares with caret}
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
```

# Logistic Regression

```{r Logistic Regression }
### 2.a. Logistic Regression without caret ----
# The glm() function is in the stats package and creates a
# generalized linear model for regression or classification.
# It can be configured to perform a logistic regression suitable for binary
# classification problems.

#### Load and split the dataset ----
library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")
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
```

# Logistic Regression with caret

```{r Logistic Regression with caret}
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

```

# Linear Discriminant Analysis

```{r Linear Discriminant Analysis }
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

```

# Linear Discriminant Analysis with caret

```{r Linear Discriminant Analysis with caret }
# Load the BCW dataset
library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")
# Define a 70:30 train:test data split of the dataset.

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


```

# Regularized Linear Regression Classification Problem with CARET

```{r Regularized Linear Regression Classification Problem with CARET }
library(readr)
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")
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

```

# Regularized Linear Regression Problem with CARET

```{r Regularized Linear Regression Regression Problem with CARET }
library(readr)
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")

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
```

# Non-Linear Algorithms

```{r Non-Linear Algorithms }

# Load necessary libraries
library(readr)
library(caret)
library(rpart)

# Load and split the dataset
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")

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

```

# Decision tree for a regression problem without CARET

```{r  Decision tree for a regression problem without CARET }
# Load the cubic_zirconia dataset
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")


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


```

# Decision tree for a classification problem with caret

```{r Decision tree for a classification problem with caret}
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




```

# Decision tree for a regression problem with CARET

```{r Decision tree for a regression problem with CARET }
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
```

# Naïve Bayes

```{r Naïve Bayes}
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
```

# Naïve Bayes Classifier for a Classification Problem with CARET

```{r Naïve Bayes Classifier for a Classification Problem with CARET}
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



```

# kNN for a classification problem without CARET's train function

```{r }
# The knn3() function is in the caret package and does not create a model.
# Instead it makes predictions from the training dataset directly.
# It can be used for classification or regression.

### 3.a. kNN for a classification problem without CARET's train function ----
#### Load and split the dataset ----
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
```

# kNN for a classification problem with CARET's train function

```{r}
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


```

# kNN for a regression problem with CARET's train function

```{r}
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

```

# Support Vector Machine for a classification problem without CARET

```{r}
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
```

# SVM Classifier for a regression problem without CARET

```{r}

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
```

# SVM Classifier for a classification problem with CARET

# The SVM with Radial Basis kernel implementation can be used with caret for classification as follows:

```{r}
#### Load and split the dataset 
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
```

# SVM Classifier for a Regression problem without CARET

# The SVM with radial basis kernel implementation can be used with caret for regression as follows:

```{r }
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
```
