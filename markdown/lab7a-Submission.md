Business Intelligence Lab Submission Markdown
================
Team Marafiki
31/10/2023

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [Load the dataset](#load-the-dataset)
- [Linear Regression](#linear-regression)
- [Linear Regression using Ordinary Least Squares with
  caret](#linear-regression-using-ordinary-least-squares-with-caret)
- [Logistic Regression](#logistic-regression)
- [Logistic Regression with caret](#logistic-regression-with-caret)
- [Linear Discriminant Analysis](#linear-discriminant-analysis)
- [Linear Discriminant Analysis with
  caret](#linear-discriminant-analysis-with-caret)
- [Regularized Linear Regression Classification Problem with
  CARET](#regularized-linear-regression-classification-problem-with-caret)
- [Regularized Linear Regression Problem with
  CARET](#regularized-linear-regression-problem-with-caret)
- [Non-Linear Algorithms](#non-linear-algorithms)
- [Decision tree for a regression problem without
  CARET](#decision-tree-for-a-regression-problem-without-caret)
- [Decision tree for a classification problem with
  caret](#decision-tree-for-a-classification-problem-with-caret)
- [Decision tree for a regression problem with
  CARET](#decision-tree-for-a-regression-problem-with-caret)
- [Naïve Bayes](#naïve-bayes)

# Student Details

<table>
<colgroup>
<col style="width: 53%" />
<col style="width: 46%" />
</colgroup>
<tbody>
<tr class="odd">
<td><strong>Student ID Numbers and Names of Group Members</strong></td>
<td><ol type="1">
<li><p>136446 - C - Mirav Bhojani</p></li>
<li><p>136788 - C - Derrick Nyaga</p></li>
<li><p>136709 - C - Jane Mugo</p></li>
<li><p>136895 - C - Wesley Wanyama</p></li>
<li><p>135399 - C - Sheilla Kavinya</p></li>
</ol></td>
</tr>
<tr class="even">
<td><strong>GitHub Classroom Group Name</strong></td>
<td><em>Team Marafiki</em></td>
</tr>
<tr class="odd">
<td><strong>Course Code</strong></td>
<td>BBT4206</td>
</tr>
<tr class="even">
<td><strong>Course Name</strong></td>
<td>Business Intelligence II</td>
</tr>
<tr class="odd">
<td><strong>Program</strong></td>
<td>Bachelor of Business Information Technology</td>
</tr>
<tr class="even">
<td><strong>Semester Duration</strong></td>
<td>21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023</td>
</tr>
</tbody>
</table>

# Setup Chunk

We start by installing all the required packages

``` r
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
```

    ## Loading required package: mlbench

``` r
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: caret

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
## MASS ----
if (require("MASS")) {
  require("MASS")
} else {
  install.packages("MASS", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: MASS

``` r
## glmnet ----
if (require("glmnet")) {
  require("glmnet")
} else {
  install.packages("glmnet", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: glmnet

    ## Loading required package: Matrix

    ## Loaded glmnet 4.1-8

``` r
## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: e1071

``` r
## kernlab ----
if (require("kernlab")) {
  require("kernlab")
} else {
  install.packages("kernlab", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: kernlab

    ## 
    ## Attaching package: 'kernlab'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     alpha

``` r
## rpart ----
if (require("rpart")) {
  require("rpart")
} else {
  install.packages("rpart", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: rpart

------------------------------------------------------------------------

**Note:** the following “*KnitR*” options have been set as the defaults
in this markdown:  
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy.opts = list(width.cutoff = 80), tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

``` r
knitr::opts_chunk$set(
    eval = TRUE,
    echo = TRUE,
    warning = FALSE,
    collapse = FALSE,
    tidy = TRUE
)
```

------------------------------------------------------------------------

**Note:** the following “*R Markdown*” options have been set as the
defaults in this markdown:

> output:  
>   
> github_document:  
> toc: yes  
> toc_depth: 4  
> fig_width: 6  
> fig_height: 4  
> df_print: default  
>   
> editor_options:  
> chunk_output_type: console

# Load the dataset

Load the BreastCancer dataset

``` r
library(readr)
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")
```

    ## Rows: 193573 Columns: 11
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (3): cut, color, clarity
    ## dbl (8): id, carat, depth, table, x, y, z, price
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# View(cz)
```

# Linear Regression

``` r
### 1.a. Linear Regression using Ordinary Least Squares without caret ---- The
### lm() function is in the stats package and creates a linear regression model
### using ordinary least squares (OLS).



# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(cz$price, p = 0.8, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
cz_model_lm <- lm(price ~ ., cz_train)

#### Display the model's details ----
print(cz_model_lm)
```

    ## 
    ## Call:
    ## lm(formula = price ~ ., data = cz_train)
    ## 
    ## Coefficients:
    ##  (Intercept)            id         carat       cutGood      cutIdeal  
    ##    1.235e+04     1.863e-05     1.440e+04     3.941e+02     5.862e+02  
    ##   cutPremium  cutVery Good        colorE        colorF        colorG  
    ##    5.206e+02     5.061e+02    -1.800e+02    -2.526e+02    -3.687e+02  
    ##       colorH        colorI        colorJ     clarityIF    claritySI1  
    ##   -8.786e+02    -1.425e+03    -2.324e+03     3.810e+03     2.513e+03  
    ##   claritySI2    clarityVS1    clarityVS2   clarityVVS1   clarityVVS2  
    ##    1.591e+03     3.382e+03     3.102e+03     3.620e+03     3.631e+03  
    ##        depth         table             x             y             z  
    ##   -1.253e+02    -3.565e+01    -1.518e+03    -6.554e+02    -1.138e+02

``` r
#### Make predictions ----
predictions <- predict(cz_model_lm, cz_test[, 1:10])

#### Display the model's evaluation metrics ---- RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))
```

    ## [1] "RMSE = 939.4796"

``` r
##### SSR ---- SSR is the sum of squared residuals (the sum of squared
##### differences between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))
```

    ## [1] "SSR = 34168944906.9737"

``` r
##### SST ---- SST is the total sum of squares (the sum of squared differences
##### between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))
```

    ## [1] "SST = 620402207184.4434"

``` r
##### R Squared ---- We then use SSR and SST to compute the value of R squared.
##### The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr/sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))
```

    ## [1] "R Squared = 0.9449"

``` r
##### MAE ---- MAE is expressed in the same units as the target variable,
##### making it easy to interpret. For example, if you are predicting the
##### amount paid in rent, and the MAE is KES. 10,000, it means, on average,
##### your model's predictions are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))
```

    ## [1] "MAE = 622.7091"

# Linear Regression using Ordinary Least Squares with caret

``` r
# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(cz$price, p = 0.8, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_lm <- train(price ~ ., data = cz_train, method = "lm", metric = "RMSE",
    preProcess = c("center", "scale"), trControl = train_control)

#### Display the model's details ----
print(cz_caret_model_lm)
```

    ## Linear Regression 
    ## 
    ## 154860 samples
    ##     10 predictor
    ## 
    ## Pre-processing: centered (24), scaled (24) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 123888, 123889, 123888, 123888, 123887 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   951.5282  0.9444138  626.5236
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
#### Make predictions ----
predictions <- predict(cz_caret_model_lm, cz_test[, 1:10])

#### Display the model's evaluation metrics ---- RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))
```

    ## [1] "RMSE = 948.3425"

``` r
##### SSR ---- SSR is the sum of squared residuals (the sum of squared
##### differences between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))
```

    ## [1] "SSR = 34816673143.6219"

``` r
##### SST ---- SST is the total sum of squares (the sum of squared differences
##### between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))
```

    ## [1] "SST = 628365760706.1505"

``` r
##### R Squared ---- We then use SSR and SST to compute the value of R squared.
##### The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr/sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))
```

    ## [1] "R Squared = 0.9446"

``` r
##### MAE ---- MAE is expressed in the same units as the target variable,
##### making it easy to interpret. For example, if you are predicting the
##### amount paid in rent, and the MAE is KES. 10,000, it means, on average,
##### your model's predictions are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))
```

    ## [1] "MAE = 625.7957"

# Logistic Regression

``` r
### 2.a. Logistic Regression without caret ---- The glm() function is in the
### stats package and creates a generalized linear model for regression or
### classification.  It can be configured to perform a logistic regression
### suitable for binary classification problems.

#### Load and split the dataset ----
library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")
```

    ## Rows: 569 Columns: 32
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (32): id, radius_mean, texture_mean, perimeter_mean, area_mean, smoothne...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(BCW)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(BCW$diagnosis, p = 0.7, list = FALSE)
BCW_train <- BCW[train_index, ]
BCW_test <- BCW[-train_index, ]

#### Train the model ----
BCW_model_glm <- glm(diagnosis ~ ., data = BCW_train, family = binomial(link = "logit"))

#### Display the model's details ----
print(BCW_model_glm)
```

    ## 
    ## Call:  glm(formula = diagnosis ~ ., family = binomial(link = "logit"), 
    ##     data = BCW_train)
    ## 
    ## Coefficients:
    ##             (Intercept)                       id              radius_mean  
    ##              -3.316e+03               -5.114e-08               -5.735e+02  
    ##            texture_mean           perimeter_mean                area_mean  
    ##               1.250e+01                8.471e+01                1.114e-01  
    ##         smoothness_mean         compactness_mean           concavity_mean  
    ##               4.914e+02               -8.701e+03                2.335e+03  
    ##   `concave points_mean`            symmetry_mean   fractal_dimension_mean  
    ##               1.521e+02                5.884e+02                1.143e+04  
    ##               radius_se               texture_se             perimeter_se  
    ##               1.947e+03                3.024e+01               -1.898e+02  
    ##                 area_se            smoothness_se           compactness_se  
    ##              -2.927e+00               -4.618e+04                1.349e+04  
    ##            concavity_se      `concave points_se`              symmetry_se  
    ##              -4.448e+03                2.469e+04                4.133e+03  
    ##    fractal_dimension_se             radius_worst            texture_worst  
    ##              -1.304e+05                7.028e+01                3.586e+00  
    ##         perimeter_worst               area_worst         smoothness_worst  
    ##               1.109e+01               -7.462e-01                4.927e+03  
    ##       compactness_worst          concavity_worst   `concave points_worst`  
    ##              -8.787e+02                3.296e+02                2.450e+02  
    ##          symmetry_worst  fractal_dimension_worst  
    ##               1.133e+02                9.556e+03  
    ## 
    ## Degrees of Freedom: 398 Total (i.e. Null);  367 Residual
    ## Null Deviance:       526.2 
    ## Residual Deviance: 3.143e-07     AIC: 64

``` r
#### Make predictions ----
probabilities <- predict(BCW_model_glm, BCW_test[, 1:31], type = "response")
print(probabilities)
```

    ##            1            2            3            4            5            6 
    ## 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 
    ##            7            8            9           10           11           12 
    ## 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 
    ##           13           14           15           16           17           18 
    ## 2.220446e-16 1.000000e+00 1.000000e+00 2.220446e-16 2.220446e-16 1.000000e+00 
    ##           19           20           21           22           23           24 
    ## 1.000000e+00 1.000000e+00 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##           25           26           27           28           29           30 
    ## 1.000000e+00 1.000000e+00 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##           31           32           33           34           35           36 
    ## 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 2.220446e-16 2.220446e-16 
    ##           37           38           39           40           41           42 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 2.220446e-16 
    ##           43           44           45           46           47           48 
    ## 1.000000e+00 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 1.000000e+00 
    ##           49           50           51           52           53           54 
    ## 2.220446e-16 1.000000e+00 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 
    ##           55           56           57           58           59           60 
    ## 1.000000e+00 1.000000e+00 2.220446e-16 2.220446e-16 1.000000e+00 1.000000e+00 
    ##           61           62           63           64           65           66 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##           67           68           69           70           71           72 
    ## 2.220446e-16 1.000000e+00 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 
    ##           73           74           75           76           77           78 
    ## 2.220446e-16 1.000000e+00 2.220446e-16 1.000000e+00 1.000000e+00 1.000000e+00 
    ##           79           80           81           82           83           84 
    ## 1.000000e+00 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 1.000000e+00 
    ##           85           86           87           88           89           90 
    ## 2.220446e-16 1.000000e+00 1.000000e+00 2.220446e-16 8.894333e-10 2.220446e-16 
    ##           91           92           93           94           95           96 
    ## 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 2.220446e-16 2.220446e-16 
    ##           97           98           99          100          101          102 
    ## 1.000000e+00 1.000000e+00 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 
    ##          103          104          105          106          107          108 
    ## 2.220446e-16 2.220446e-16 1.000000e+00 1.000000e+00 2.220446e-16 7.483102e-03 
    ##          109          110          111          112          113          114 
    ## 1.000000e+00 2.220446e-16 2.220446e-16 2.220446e-16 1.000000e+00 1.000000e+00 
    ##          115          116          117          118          119          120 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 1.000000e+00 1.000000e+00 2.220446e-16 
    ##          121          122          123          124          125          126 
    ## 2.220446e-16 1.000000e+00 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 
    ##          127          128          129          130          131          132 
    ## 1.000000e+00 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##          133          134          135          136          137          138 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 1.000000e+00 2.220446e-16 2.220446e-16 
    ##          139          140          141          142          143          144 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##          145          146          147          148          149          150 
    ## 1.000000e+00 1.000000e+00 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##          151          152          153          154          155          156 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##          157          158          159          160          161          162 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 
    ##          163          164          165          166          167          168 
    ## 2.220446e-16 2.220446e-16 2.220446e-16 2.220446e-16 1.000000e+00 1.000000e+00 
    ##          169          170 
    ## 1.000000e+00 2.220446e-16

``` r
predictions <- ifelse(probabilities > 0.5, 1, 0)
print(predictions)
```

    ##   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
    ##   1   1   1   1   1   1   1   1   1   1   1   1   0   1   1   0   0   1   1   1 
    ##  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
    ##   0   0   0   0   1   1   0   0   0   0   1   1   1   1   0   0   0   0   0   1 
    ##  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 
    ##   0   0   1   0   0   1   0   1   0   1   0   0   1   0   1   1   0   0   1   1 
    ##  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
    ##   0   0   0   0   0   0   0   1   0   0   1   0   0   1   0   1   1   1   1   0 
    ##  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 
    ##   0   0   0   1   0   1   1   0   0   0   0   0   1   0   0   0   1   1   0   0 
    ## 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 
    ##   1   0   0   0   1   1   0   0   1   0   0   0   1   1   0   0   0   1   1   0 
    ## 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 
    ##   0   1   0   0   1   0   1   0   0   0   0   0   0   0   0   1   0   0   0   0 
    ## 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 
    ##   0   0   0   0   1   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
    ## 161 162 163 164 165 166 167 168 169 170 
    ##   0   0   0   0   0   0   1   1   1   0

``` r
#### Display the model's evaluation metrics ----
table(predictions, BCW_test$diagnosis)
```

    ##            
    ## predictions   0   1
    ##           0 103   4
    ##           1   3  60

# Logistic Regression with caret

``` r
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
```

    ## Generalized Linear Model 
    ## 
    ## 399 samples
    ##  31 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## Pre-processing: centered (31), scaled (31) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 320, 319, 319, 319, 319 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9173734  0.8279461

``` r
# Make predictions
predictions <- predict(BCW_caret_model_logistic, newdata = BCW_test)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 106   5
    ##          1   2  57
    ##                                          
    ##                Accuracy : 0.9588         
    ##                  95% CI : (0.917, 0.9833)
    ##     No Information Rate : 0.6353         
    ##     P-Value [Acc > NIR] : <2e-16         
    ##                                          
    ##                   Kappa : 0.9102         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.4497         
    ##                                          
    ##             Sensitivity : 0.9815         
    ##             Specificity : 0.9194         
    ##          Pos Pred Value : 0.9550         
    ##          Neg Pred Value : 0.9661         
    ##              Prevalence : 0.6353         
    ##          Detection Rate : 0.6235         
    ##    Detection Prevalence : 0.6529         
    ##       Balanced Accuracy : 0.9504         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

``` r
# Create a confusion matrix plot
fourfoldplot(confusion_matrix$table, color = c("grey", "lightblue"),
             main = "Confusion Matrix")
```

![](lab7a-Submission_files/figure-gfm/Logistic%20Regression%20with%20caret-1.png)<!-- -->

# Linear Discriminant Analysis

``` r
# Load the BCW dataset
library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")
```

    ## Rows: 569 Columns: 32
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (32): id, radius_mean, texture_mean, perimeter_mean, area_mean, smoothne...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
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
```

    ##    
    ##       0   1
    ##   0 102   8
    ##   1   1  59

``` r
# Create a confusion matrix plot
fourfoldplot(confusion_matrix, color = c("grey", "lightblue"), main = "Confusion Matrix")
```

![](lab7a-Submission_files/figure-gfm/Linear%20Discriminant%20Analysis-1.png)<!-- -->

# Linear Discriminant Analysis with caret

``` r
# Load the BCW dataset
library(readr)
BCW <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/BCW.csv")
```

    ## Rows: 569 Columns: 32
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (32): id, radius_mean, texture_mean, perimeter_mean, area_mean, smoothne...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
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
lda_model <- train(diagnosis ~ ., data = BCW_train, method = "lda", metric = "Accuracy",
    preProcess = c("center", "scale"), trControl = train_control)

# Display the model's details
print(lda_model)
```

    ## Linear Discriminant Analysis 
    ## 
    ## 399 samples
    ##  31 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## Pre-processing: centered (31), scaled (31) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 398, 398, 398, 398, 398, 398, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.9598997  0.9112298

``` r
# Make predictions
predictions <- predict(lda_model, newdata = BCW_test)

# Display the model's evaluation metrics (confusion matrix)
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 102   8
    ##          1   1  59
    ##                                           
    ##                Accuracy : 0.9471          
    ##                  95% CI : (0.9019, 0.9755)
    ##     No Information Rate : 0.6059          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.8871          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0455          
    ##                                           
    ##             Sensitivity : 0.9903          
    ##             Specificity : 0.8806          
    ##          Pos Pred Value : 0.9273          
    ##          Neg Pred Value : 0.9833          
    ##              Prevalence : 0.6059          
    ##          Detection Rate : 0.6000          
    ##    Detection Prevalence : 0.6471          
    ##       Balanced Accuracy : 0.9354          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

# Regularized Linear Regression Classification Problem with CARET

``` r
library(readr)
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")
```

    ## Rows: 193573 Columns: 11
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (3): cut, color, clarity
    ## dbl (8): id, carat, depth, table, x, y, z, price
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(cz$price, p = 0.7, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_glmnet <- train(price ~ ., data = cz_train, method = "glmnet", metric = "RMSE",
    preProcess = c("center", "scale"), trControl = train_control)

#### Display the model's details ----
print(cz_caret_model_glmnet)
```

    ## glmnet 
    ## 
    ## 135502 samples
    ##     10 predictor
    ## 
    ## Pre-processing: centered (24), scaled (24) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 108403, 108401, 108402, 108401, 108401 
    ## Resampling results across tuning parameters:
    ## 
    ##   alpha  lambda      RMSE       Rsquared   MAE      
    ##   0.10     7.608949   955.5082  0.9438846   631.3833
    ##   0.10    76.089491  1069.0881  0.9302878   751.7066
    ##   0.10   760.894907  1425.0378  0.8808925  1014.7576
    ##   0.55     7.608949   963.0710  0.9429909   632.2025
    ##   0.55    76.089491  1098.3915  0.9264494   766.6685
    ##   0.55   760.894907  1550.2812  0.8711678  1019.5440
    ##   1.00     7.608949   968.6182  0.9423245   631.6394
    ##   1.00    76.089491  1124.0642  0.9230222   780.4954
    ##   1.00   760.894907  1538.3045  0.8900801   972.9295
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were alpha = 0.1 and lambda = 7.608949.

``` r
#### Make predictions ----
predictions <- predict(cz_caret_model_glmnet, cz_test[, 1:10])

#### Display the model's evaluation metrics ---- RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))
```

    ## [1] "RMSE = 963.8738"

``` r
##### SSR ---- SSR is the sum of squared residuals (the sum of squared
##### differences between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))
```

    ## [1] "SSR = 53951016619.5280"

``` r
##### SST ---- SST is the total sum of squares (the sum of squared differences
##### between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))
```

    ## [1] "SST = 947156394119.3772"

``` r
##### R Squared ---- We then use SSR and SST to compute the value of R squared.
##### The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr/sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))
```

    ## [1] "R Squared = 0.9430"

``` r
##### MAE ---- MAE is expressed in the same units as the target variable,
##### making it easy to interpret. For example, if you are predicting the
##### amount paid in rent, and the MAE is KES. 10,000, it means, on average,
##### your model's predictions are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))
```

    ## [1] "MAE = 638.4033"

# Regularized Linear Regression Problem with CARET

``` r
library(readr)
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")
```

    ## Rows: 193573 Columns: 11
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (3): cut, color, clarity
    ## dbl (8): id, carat, depth, table, x, y, z, price
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(cz$price, p = 0.7, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
cz_caret_model_glmnet <- train(price ~ ., data = cz_train, method = "glmnet", metric = "RMSE",
    preProcess = c("center", "scale"), trControl = train_control)

#### Display the model's details ----
print(cz_caret_model_glmnet)
```

    ## glmnet 
    ## 
    ## 135502 samples
    ##     10 predictor
    ## 
    ## Pre-processing: centered (24), scaled (24) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 108401, 108402, 108401, 108402, 108402 
    ## Resampling results across tuning parameters:
    ## 
    ##   alpha  lambda      RMSE       Rsquared   MAE      
    ##   0.10     7.613204   962.4825  0.9431617   636.6158
    ##   0.10    76.132038  1073.7675  0.9297812   755.7383
    ##   0.10   761.320377  1427.3207  0.8806814  1016.7262
    ##   0.55     7.613204   970.6084  0.9421961   637.6253
    ##   0.55    76.132038  1102.7107  0.9259865   769.9848
    ##   0.55   761.320377  1552.8763  0.8708935  1022.0399
    ##   1.00     7.613204   975.2553  0.9416370   637.4796
    ##   1.00    76.132038  1128.2753  0.9225658   783.4879
    ##   1.00   761.320377  1541.3557  0.8897247   975.5625
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were alpha = 0.1 and lambda = 7.613204.

``` r
#### Make predictions ----
predictions <- predict(cz_caret_model_glmnet, cz_test[, 1:10])

#### Display the model's evaluation metrics ---- RMSE ----
rmse <- sqrt(mean((cz_test$price - predictions)^2))
print(paste("RMSE =", sprintf(rmse, fmt = "%#.4f")))
```

    ## [1] "RMSE = 949.1727"

``` r
##### SSR ---- SSR is the sum of squared residuals (the sum of squared
##### differences between observed and predicted values)
ssr <- sum((cz_test$price - predictions)^2)
print(paste("SSR =", sprintf(ssr, fmt = "%#.4f")))
```

    ## [1] "SSR = 52317835743.0392"

``` r
##### SST ---- SST is the total sum of squares (the sum of squared differences
##### between observed values and their mean)
sst <- sum((cz_test$price - mean(cz_test$price))^2)
print(paste("SST =", sprintf(sst, fmt = "%#.4f")))
```

    ## [1] "SST = 943765614815.2786"

``` r
##### R Squared ---- We then use SSR and SST to compute the value of R squared.
##### The closer the R squared value is to 1, the better the model.
r_squared <- 1 - (ssr/sst)
print(paste("R Squared =", sprintf(r_squared, fmt = "%#.4f")))
```

    ## [1] "R Squared = 0.9446"

``` r
##### MAE ---- MAE is expressed in the same units as the target variable,
##### making it easy to interpret. For example, if you are predicting the
##### amount paid in rent, and the MAE is KES. 10,000, it means, on average,
##### your model's predictions are off by about KES. 10,000.
absolute_errors <- abs(predictions - cz_test$price)
mae <- mean(absolute_errors)
print(paste("MAE =", sprintf(mae, fmt = "%#.4f")))
```

    ## [1] "MAE = 632.1629"

# Non-Linear Algorithms

``` r
# Load necessary libraries
library(readr)
library(caret)
library(rpart)

# Load and split the dataset
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")
```

    ## Rows: 193573 Columns: 11
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (3): cut, color, clarity
    ## dbl (8): id, carat, depth, table, x, y, z, price
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# Define a 70:30 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(cz$cut, p = 0.7, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the model
cz_model_rpart <- rpart(cut ~ ., data = cz_train)

# Display the model's details
print(cz_model_rpart)
```

    ## n= 135503 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 135503 70785 Ideal (0.01 0.06 0.48 0.26 0.19)  
    ##    2) table< 57.15 80475 17885 Ideal (0.007 0.04 0.78 0.038 0.14)  
    ##      4) depth< 62.75 68925  8606 Ideal (0.0003 0.0022 0.88 0.04 0.083) *
    ##      5) depth>=62.75 11550  6209 Very Good (0.047 0.27 0.2 0.028 0.46)  
    ##       10) depth>=63.55 2888   921 Good (0.18 0.68 0.013 0.001 0.12) *
    ##       11) depth< 63.55 8662  3670 Very Good (0.00081 0.13 0.26 0.037 0.58) *
    ##    3) table>=57.15 55028 23141 Premium (0.016 0.089 0.039 0.58 0.28)  
    ##      6) depth>=63.05 6282  3334 Good (0.1 0.47 0.0014 0.0024 0.43)  
    ##       12) depth>=63.55 2961   923 Good (0.21 0.69 0.0017 0.0034 0.095) *
    ##       13) depth< 63.55 3321   928 Very Good (0.0027 0.27 0.0012 0.0015 0.72) *
    ##      7) depth< 63.05 48746 16874 Premium (0.0045 0.04 0.043 0.65 0.26) *

``` r
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
# confusion_matrix <- confusionMatrix(predictions, cz_test$cut)
# print(confusion_matrix)
```

# Decision tree for a regression problem without CARET

``` r
# Load the cubic_zirconia dataset
cz <- read_csv("/Users/nyagaderrick/Developer/BBT4206-R-Lab7of15-AlgorithmSelection-team-marafiki/data/cubic_zirconia.csv")
```

    ## Rows: 193573 Columns: 11
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (3): cut, color, clarity
    ## dbl (8): id, carat, depth, table, x, y, z, price
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# Define an 80:20 train:test data split of the dataset.
set.seed(7)  # For reproducibility
train_index <- createDataPartition(cz$cut, p = 0.8, list = FALSE)
cz_train <- cz[train_index, ]
cz_test <- cz[-train_index, ]

# Train the model for a classification problem (e.g., predicting 'cut')
cz_model_rpart <- rpart(cut ~ ., data = cz_train, method = "class")

# Display the model's details
print(cz_model_rpart)
```

    ## n= 154860 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 154860 80896 Ideal (0.01 0.06 0.48 0.26 0.19)  
    ##    2) table< 57.15 91756 20298 Ideal (0.007 0.04 0.78 0.038 0.14)  
    ##      4) depth< 62.75 78569  9753 Ideal (0.00034 0.0022 0.88 0.039 0.082) *
    ##      5) depth>=62.75 13187  7084 Very Good (0.046 0.26 0.2 0.028 0.46)  
    ##       10) depth>=63.55 3257  1057 Good (0.19 0.68 0.013 0.0012 0.12) *
    ##       11) depth< 63.55 9930  4233 Very Good (0.00081 0.13 0.26 0.037 0.57) *
    ##    3) table>=57.15 63104 26622 Premium (0.015 0.09 0.04 0.58 0.28)  
    ##      6) depth>=63.05 7278  3883 Good (0.1 0.47 0.0019 0.0022 0.43)  
    ##       12) depth>=63.55 3418  1057 Good (0.21 0.69 0.0018 0.0029 0.092) *
    ##       13) depth< 63.55 3860  1060 Very Good (0.0031 0.27 0.0021 0.0016 0.73) *
    ##      7) depth< 63.05 55826 19360 Premium (0.0043 0.041 0.045 0.65 0.26) *

``` r
# Make predictions
predictions <- predict(cz_model_rpart, newdata = cz_test, type = "class")

# Display the model's evaluation metrics (classification report)
confusion_matrix <- table(predictions, cz_test$cut)
print(confusion_matrix)
```

    ##            
    ## predictions  Fair  Good Ideal Premium Very Good
    ##   Fair          0     0     0       0         0
    ##   Good        328  1136    12       5       179
    ##   Ideal         7    39 17262     772      1560
    ##   Premium      62   558   613    9114      3662
    ##   Very Good     7   591   603      91      2112

``` r
# Calculate other classification metrics (e.g., accuracy)
accuracy <- sum(diag(confusion_matrix))/sum(confusion_matrix)
print(paste("Accuracy =", sprintf(accuracy, fmt = "%#.4f")))
```

    ## [1] "Accuracy = 0.7652"

# Decision tree for a classification problem with caret

``` r
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
BCW_caret_model_tree <- train(diagnosis ~ ., data = BCW_train, method = "rpart",
    metric = "Accuracy", trControl = trainControl(method = "cv", number = 5))

# Display the model's details
print(BCW_caret_model_tree)
```

    ## CART 
    ## 
    ## 399 samples
    ##  31 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 319, 319, 319, 320, 319 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa    
    ##   0.02348993  0.9272785  0.8435282
    ##   0.08724832  0.9096835  0.8039332
    ##   0.79865772  0.7796835  0.4476208
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.02348993.

``` r
# Make predictions
predictions <- predict(BCW_caret_model_tree, newdata = BCW_test)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 105  14
    ##          1   2  49
    ##                                           
    ##                Accuracy : 0.9059          
    ##                  95% CI : (0.8517, 0.9452)
    ##     No Information Rate : 0.6294          
    ##     P-Value [Acc > NIR] : < 2e-16         
    ##                                           
    ##                   Kappa : 0.79            
    ##                                           
    ##  Mcnemar's Test P-Value : 0.00596         
    ##                                           
    ##             Sensitivity : 0.9813          
    ##             Specificity : 0.7778          
    ##          Pos Pred Value : 0.8824          
    ##          Neg Pred Value : 0.9608          
    ##              Prevalence : 0.6294          
    ##          Detection Rate : 0.6176          
    ##    Detection Prevalence : 0.7000          
    ##       Balanced Accuracy : 0.8795          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

# Decision tree for a regression problem with CARET

``` r
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
BCW_caret_model_tree <- train(diagnosis ~ ., data = BCW_train, method = "rpart",
    metric = "Accuracy", trControl = trainControl(method = "cv", number = 5))

# Display the model's details
print(BCW_caret_model_tree)
```

    ## CART 
    ## 
    ## 399 samples
    ##  31 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 319, 319, 319, 320, 319 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa    
    ##   0.02348993  0.9272785  0.8435282
    ##   0.08724832  0.9096835  0.8039332
    ##   0.79865772  0.7796835  0.4476208
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.02348993.

``` r
# Make predictions
predictions <- predict(BCW_caret_model_tree, newdata = BCW_test)

# Display the model's evaluation metrics
confusion_matrix <- confusionMatrix(predictions, BCW_test$diagnosis)
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 105  14
    ##          1   2  49
    ##                                           
    ##                Accuracy : 0.9059          
    ##                  95% CI : (0.8517, 0.9452)
    ##     No Information Rate : 0.6294          
    ##     P-Value [Acc > NIR] : < 2e-16         
    ##                                           
    ##                   Kappa : 0.79            
    ##                                           
    ##  Mcnemar's Test P-Value : 0.00596         
    ##                                           
    ##             Sensitivity : 0.9813          
    ##             Specificity : 0.7778          
    ##          Pos Pred Value : 0.8824          
    ##          Neg Pred Value : 0.9608          
    ##              Prevalence : 0.6294          
    ##          Detection Rate : 0.6176          
    ##    Detection Prevalence : 0.7000          
    ##       Balanced Accuracy : 0.8795          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

``` r
# Calculate and display Mean Absolute Error (MAE)
absolute_errors <- abs(as.numeric(predictions) - as.numeric(BCW_test$diagnosis))
mae <- mean(absolute_errors)
print(paste("Mean Absolute Error (MAE) =", sprintf(mae, fmt = "%#.4f")))
```

    ## [1] "Mean Absolute Error (MAE) = 0.0941"

# Naïve Bayes

``` r
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

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 101  11
    ##          1   6  52
    ##                                           
    ##                Accuracy : 0.9             
    ##                  95% CI : (0.8447, 0.9407)
    ##     No Information Rate : 0.6294          
    ##     P-Value [Acc > NIR] : 1.01e-15        
    ##                                           
    ##                   Kappa : 0.7821          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.332           
    ##                                           
    ##             Sensitivity : 0.9439          
    ##             Specificity : 0.8254          
    ##          Pos Pred Value : 0.9018          
    ##          Neg Pred Value : 0.8966          
    ##              Prevalence : 0.6294          
    ##          Detection Rate : 0.5941          
    ##    Detection Prevalence : 0.6588          
    ##       Balanced Accuracy : 0.8847          
    ##                                           
    ##        'Positive' Class : 0               
    ## 


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
