---
editor_options: 
  markdown: 
    wrap: 72
---

# Business Intelligence Lab Submission Markdown

Team Marafiki 03/11/2023

-   [Student Details](#student-details)
    -   [Installing Packages Required](#installing-packages-required)
    -   [Loading the dataset](#loading-the-dataset)
    -   [Transform the dataset](#transform-the-dataset)
    -   [Split the dataset](#split-the-dataset)
    -   [Basic EDA](#basic-eda)
    -   [Creating the association
        rules](#creating-the-association-rules)
    -   [Finding Specific Rules](#finding-specific-rules)
    -   [Visualizing the Rules](#visualizing-the-rules)

# Student Details {#student-details}

+-------------------------------------+--------------------------------+
| **Student ID Numbers and Names of   | 1.  136446 - C - Mirav Bhojani |
| Group Members**                     |                                |
|                                     | 2.  136788 - C - Derrick Nyaga |
|                                     |                                |
|                                     | 3.  136709 - C - Jane Mugo     |
|                                     |                                |
|                                     | 4.  136895 - C - Wesley        |
|                                     |     Wanyama                    |
|                                     |                                |
|                                     | 5.  135399 - C - Sheilla       |
|                                     |     Kavinya                    |
+-------------------------------------+--------------------------------+
| **GitHub Classroom Group Name**     | *Team Marafiki*                |
+-------------------------------------+--------------------------------+
| **Course Code**                     | BBT4206                        |
+-------------------------------------+--------------------------------+
| **Course Name**                     | Business Intelligence II       |
+-------------------------------------+--------------------------------+
| **Program**                         | Bachelor of Business           |
|                                     | Information Technology         |
+-------------------------------------+--------------------------------+
| **Semester Duration**               | 21^st^ August 2023 to 28^th^   |
|                                     | November 2023                  |
+-------------------------------------+--------------------------------+

## Installing Packages Required {#installing-packages-required}

``` r
if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
## arules ----
if (require("arules")) {
  require("arules")
} else {
  install.packages("arules", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## arulesViz ----
if (require("arulesViz")) {
  require("arulesViz")
} else {
  install.packages("arulesViz", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## tidyverse ----
if (require("tidyverse")) {
  require("tidyverse")
} else {
  install.packages("tidyverse", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readxl ----
if (require("readxl")) {
  require("readxl")
} else {
  install.packages("readxl", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## knitr ----
if (require("knitr")) {
  require("knitr")
} else {
  install.packages("knitr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## lubridate ----
if (require("lubridate")) {
  require("lubridate")
} else {
  install.packages("lubridate", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## plyr ----
if (require("plyr")) {
  require("plyr")
} else {
  install.packages("plyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## RColorBrewer ----
if (require("RColorBrewer")) {
  require("RColorBrewer")
} else {
  install.packages("RColorBrewer", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

## Loading the dataset {#loading-the-dataset}

The dataset used contains information on transactions of various items
together with their invoice details.

``` r
itemslist <- read_excel("data/Assignment-1_Data.xlsx")
```

```         
## Warning: Expecting numeric in A288774 / R288774C1: got 'A563185'

## Warning: Expecting numeric in A288775 / R288775C1: got 'A563186'

## Warning: Expecting numeric in A288776 / R288776C1: got 'A563187'
```

## Transform the dataset {#transform-the-dataset}

``` r
### Handle missing values ----
# Are there missing values in the dataset?
any_na(itemslist)
```

```         
## [1] TRUE
```

``` r
# How many?
n_miss(itemslist)
```

```         
## [1] 135499
```

``` r
# What is the proportion of missing data in the entire dataset?
prop_miss(itemslist)
```

```         
## [1] 0.03707783
```

``` r
# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(itemslist)
```

```         
## # A tibble: 7 × 3
##   variable   n_miss  pct_miss
##   <chr>       <int>     <dbl>
## 1 CustomerID 134041 25.7     
## 2 Itemname     1455  0.279   
## 3 BillNo          3  0.000575
## 4 Quantity        0  0       
## 5 Date            0  0       
## 6 Price           0  0       
## 7 Country         0  0
```

``` r
# Which variables contain the most missing values?
gg_miss_var(itemslist)
```

![](lab7C-submission-markdown_files/figure-gfm/data%20transformation-1.png)<!-- -->

``` r
# Which combinations of variables are missing together?
gg_miss_upset(itemslist)
```

![](lab7C-submission-markdown_files/figure-gfm/data%20transformation-2.png)<!-- -->

``` r
#### OPTION 1: Remove the observations with missing values ----
itemslist_removed_obs <- itemslist %>% dplyr::filter(complete.cases(.))

# We end up with 388,023 observations to create the association rules
# instead of the initial 522,064 observations.
dim(itemslist_removed_obs)
```

```         
## [1] 388023      7
```

``` r
# Are there missing values in the dataset?
any_na(itemslist_removed_obs)
```

```         
## [1] FALSE
```

``` r
#### OPTION 2: Remove the variables with missing values ----
# The `CustomerID` variable will not be used to create association rules.
# We will use the `BillNo` instead.
itemslist_removed_vars <-
  itemslist %>% dplyr::select(-CustomerID)

dim(itemslist_removed_vars)
```

```         
## [1] 522064      6
```

``` r
# Are there missing values in the dataset?
any_na(itemslist_removed_vars)
```

```         
## [1] TRUE
```

``` r
# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(itemslist_removed_vars)
```

```         
## # A tibble: 6 × 3
##   variable n_miss pct_miss
##   <chr>     <int>    <dbl>
## 1 Itemname   1455 0.279   
## 2 BillNo        3 0.000575
## 3 Quantity      0 0       
## 4 Date          0 0       
## 5 Price         0 0       
## 6 Country       0 0
```

``` r
# We now remove the observations that do not have a value for the description
# variable.
itemslist_removed_vars_obs <- itemslist_removed_vars %>% filter(complete.cases(.))

# We end up with 520,606 observations to create the association rules
# instead of the initial 522,064 observations.
# This is better than OPTION 1 which resulted in 388,023 observations to
# create the association rules.
dim(itemslist_removed_vars_obs)
```

```         
## [1] 520606      6
```

``` r
## Identify categorical variables ----
# Ensure the country is recorded as categorical data
itemslist_removed_vars_obs %>% mutate(Country = as.factor(Country))
```

```         
## # A tibble: 520,606 × 6
##    BillNo Itemname                    Quantity Date                Price Country
##     <dbl> <chr>                          <dbl> <dttm>              <dbl> <fct>  
##  1 536365 WHITE HANGING HEART T-LIGH…        6 2010-12-01 08:26:00  2.55 United…
##  2 536365 WHITE METAL LANTERN                6 2010-12-01 08:26:00  3.39 United…
##  3 536365 CREAM CUPID HEARTS COAT HA…        8 2010-12-01 08:26:00  2.75 United…
##  4 536365 KNITTED UNION FLAG HOT WAT…        6 2010-12-01 08:26:00  3.39 United…
##  5 536365 RED WOOLLY HOTTIE WHITE HE…        6 2010-12-01 08:26:00  3.39 United…
##  6 536365 SET 7 BABUSHKA NESTING BOX…        2 2010-12-01 08:26:00  7.65 United…
##  7 536365 GLASS STAR FROSTED T-LIGHT…        6 2010-12-01 08:26:00  4.25 United…
##  8 536366 HAND WARMER UNION JACK             6 2010-12-01 08:28:00  1.85 United…
##  9 536366 HAND WARMER RED POLKA DOT          6 2010-12-01 08:28:00  1.85 United…
## 10 536367 ASSORTED COLOUR BIRD ORNAM…       32 2010-12-01 08:34:00  1.69 United…
## # ℹ 520,596 more rows
```

``` r
# Also ensure that the Itemname is recorded
# as categorical data
itemslist_removed_vars_obs %>% mutate(Itemname = as.factor(Itemname))
```

```         
## # A tibble: 520,606 × 6
##    BillNo Itemname                    Quantity Date                Price Country
##     <dbl> <fct>                          <dbl> <dttm>              <dbl> <chr>  
##  1 536365 WHITE HANGING HEART T-LIGH…        6 2010-12-01 08:26:00  2.55 United…
##  2 536365 WHITE METAL LANTERN                6 2010-12-01 08:26:00  3.39 United…
##  3 536365 CREAM CUPID HEARTS COAT HA…        8 2010-12-01 08:26:00  2.75 United…
##  4 536365 KNITTED UNION FLAG HOT WAT…        6 2010-12-01 08:26:00  3.39 United…
##  5 536365 RED WOOLLY HOTTIE WHITE HE…        6 2010-12-01 08:26:00  3.39 United…
##  6 536365 SET 7 BABUSHKA NESTING BOX…        2 2010-12-01 08:26:00  7.65 United…
##  7 536365 GLASS STAR FROSTED T-LIGHT…        6 2010-12-01 08:26:00  4.25 United…
##  8 536366 HAND WARMER UNION JACK             6 2010-12-01 08:28:00  1.85 United…
##  9 536366 HAND WARMER RED POLKA DOT          6 2010-12-01 08:28:00  1.85 United…
## 10 536367 ASSORTED COLOUR BIRD ORNAM…       32 2010-12-01 08:34:00  1.69 United…
## # ℹ 520,596 more rows
```

``` r
str(itemslist_removed_vars_obs)
```

```         
## tibble [520,606 × 6] (S3: tbl_df/tbl/data.frame)
##  $ BillNo  : num [1:520606] 536365 536365 536365 536365 536365 ...
##  $ Itemname: chr [1:520606] "WHITE HANGING HEART T-LIGHT HOLDER" "WHITE METAL LANTERN" "CREAM CUPID HEARTS COAT HANGER" "KNITTED UNION FLAG HOT WATER BOTTLE" ...
##  $ Quantity: num [1:520606] 6 6 8 6 6 2 6 6 6 32 ...
##  $ Date    : POSIXct[1:520606], format: "2010-12-01 08:26:00" "2010-12-01 08:26:00" ...
##  $ Price   : num [1:520606] 2.55 3.39 2.75 3.39 3.39 7.65 4.25 1.85 1.85 1.69 ...
##  $ Country : chr [1:520606] "United Kingdom" "United Kingdom" "United Kingdom" "United Kingdom" ...
```

``` r
dim(itemslist_removed_vars_obs)
```

```         
## [1] 520606      6
```

``` r
head(itemslist_removed_vars_obs)
```

```         
## # A tibble: 6 × 6
##   BillNo Itemname                     Quantity Date                Price Country
##    <dbl> <chr>                           <dbl> <dttm>              <dbl> <chr>  
## 1 536365 WHITE HANGING HEART T-LIGHT…        6 2010-12-01 08:26:00  2.55 United…
## 2 536365 WHITE METAL LANTERN                 6 2010-12-01 08:26:00  3.39 United…
## 3 536365 CREAM CUPID HEARTS COAT HAN…        8 2010-12-01 08:26:00  2.75 United…
## 4 536365 KNITTED UNION FLAG HOT WATE…        6 2010-12-01 08:26:00  3.39 United…
## 5 536365 RED WOOLLY HOTTIE WHITE HEA…        6 2010-12-01 08:26:00  3.39 United…
## 6 536365 SET 7 BABUSHKA NESTING BOXES        2 2010-12-01 08:26:00  7.65 United…
```

``` r
## Record the date and time variables in the correct format ----
# Ensure that Date is stored in the correct date format.
# We can separate the date and the time into 2 different variables.
itemslist_removed_vars_obs$transaction_date <-
  as.Date(itemslist_removed_vars_obs$Date)

# Extract time from Date and store it in another variable
itemslist_removed_vars_obs$transaction_time <-
  format(itemslist_removed_vars_obs$Date, "%H:%M:%S")

## Record the BillNo in the correct format (numeric) ----
# Convert BillNo into numeric
itemslist_removed_vars_obs$bill_no <-
  as.numeric(as.character(itemslist_removed_vars_obs$BillNo))


# We then remove the duplicate variables that we do not need
# (BillNo and Date) and we also remove all commas to make it easier
# to identify individual products (some products have commas in their names).
itemslist_removed_vars_obs <-
  itemslist_removed_vars_obs %>%
  select(-BillNo, -Date) %>%
  mutate_all(~str_replace_all(., ",", " "))

# The pre-processed data frame now has 520,606 observations and 8 variables.
dim(itemslist_removed_vars_obs)
```

```         
## [1] 520606      7
```

``` r
View(itemslist_removed_vars_obs)


itemslist_removed_vars_obs <-
  read.csv(file = "data/itemslist_data_before_single_transaction_format.csv")
```

## Split the dataset {#split-the-dataset}

ddply is used to split a data frame, apply a function to the split data,
and then return the result back in a data frame.

``` r
transaction_data <-
  plyr::ddply(itemslist_removed_vars_obs,
              c("bill_no", "transaction_date"),
              function(df1) {
                paste(df1$Description, collapse = ",")
              }
  )

View(transaction_data)

transaction_data <-
  transaction_data %>%
  dplyr::select("items" = V1)

View(transaction_data)

## Save the transactions in CSV format ----
write.csv(transaction_data,
          "data/itemslist_basket_format.csv",
          quote = FALSE, row.names = FALSE)

# We can now, finally, read the basket format transaction data as a
# transaction object.
tr <-
  read.transactions("data/itemslist_data_before_single_transaction_format.csv",
                    format = "basket",
                    header = TRUE,
                    rm.duplicates = TRUE,
                    sep = ","
  )
```

```         
## distribution of transactions with duplicates:
##  2 
## 12
```

## Basic EDA {#basic-eda}

This code segment is used to create a customized item frequency plot
using the "arules" package. The item frequency plot displays the
frequency of items in the dataset.

``` r
# Specify the limits for the x-axis 
custom_xlim <- c(0, 100)  

# Specify the limits for the y-axis 
custom_ylim <- c(0, 500)

itemFrequencyPlot(tr, topN = 10, type = "absolute",
                  col = brewer.pal(8, "Pastel2"),
                  main = "Absolute Item Frequency Plot",
                  horiz = TRUE,
                  xlim = custom_xlim,  # Set x-axis limits
                  ylim = custom_ylim,
                  mai = c(1.5, 1.5, 1.5, 1.5))
```

![](lab7C-submission-markdown_files/figure-gfm/Basic%20EDA-1.png)<!-- -->

## Creating the association rules {#creating-the-association-rules}

This code chunk performs association rule mining using the "arules"
package, filters out redundant rules and saves the resulting association
rules to a CSV file.

``` r
association_rules <- apriori(tr,
                             parameter = list(support = 0.01,
                                              confidence = 0.8,
                                              maxlen = 10))
```

```         
## Apriori
## 
## Parameter specification:
##  confidence minval smax arem  aval originalSupport maxtime support minlen
##         0.8    0.1    1 none FALSE            TRUE       5    0.01      1
##  maxlen target  ext
##      10  rules TRUE
## 
## Algorithmic control:
##  filter tree heap memopt load sort verbose
##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
## 
## Absolute minimum support count: 5206 
## 
## set item appearances ...[0 item(s)] done [0.00s].
## set transactions ...[27319 item(s), 520606 transaction(s)] done [0.95s].
## sorting and recoding items ... [42 item(s)] done [0.01s].
## creating transaction tree ... done [0.27s].
## checking subsets of size 1 2 3 done [0.00s].
## writing ... [49 rule(s)] done [0.00s].
## creating S4 object  ... done [0.04s].
```

``` r
#Print the association rules ----
summary(association_rules)
```

```         
## set of 49 rules
## 
## rule length distribution (lhs + rhs):sizes
##  1  2  3 
##  1 34 14 
## 
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   1.000   2.000   2.000   2.265   3.000   3.000 
## 
## summary of quality measures:
##     support          confidence        coverage            lift       
##  Min.   :0.01055   Min.   :0.8202   Min.   :0.01068   Min.   :0.8783  
##  1st Qu.:0.01315   1st Qu.:0.8995   1st Qu.:0.01410   1st Qu.:0.9632  
##  Median :0.02183   Median :0.9306   Median :0.02427   Median :0.9965  
##  Mean   :0.05475   Mean   :0.9358   Mean   :0.05848   Mean   :1.0021  
##  3rd Qu.:0.03813   3rd Qu.:0.9866   3rd Qu.:0.04414   3rd Qu.:1.0565  
##  Max.   :0.93384   Max.   :1.0000   Max.   :1.00000   Max.   :1.0708  
##      count       
##  Min.   :  5492  
##  1st Qu.:  6844  
##  Median : 11367  
##  Mean   : 28504  
##  3rd Qu.: 19850  
##  Max.   :486164  
## 
## mining info:
##  data ntransactions support confidence
##    tr        520606    0.01        0.8
##                                                                                 call
##  apriori(data = tr, parameter = list(support = 0.01, confidence = 0.8, maxlen = 10))
```

``` r
inspect(association_rules)
```

```         
##      lhs           rhs              support    confidence coverage   lift     
## [1]  {}         => {United Kingdom} 0.93384248 0.9338425  1.00000000 1.0000000
## [2]  {7.95}     => {United Kingdom} 0.01189959 0.9251792  0.01286193 0.9907230
## [3]  {5.79}     => {United Kingdom} 0.01314622 1.0000000  0.01314622 1.0708444
## [4]  {4.25}     => {United Kingdom} 0.01240093 0.9247959  0.01340937 0.9903125
## [5]  {0.55}     => {United Kingdom} 0.01287730 0.9131027  0.01410280 0.9777909
## [6]  {5.95}     => {United Kingdom} 0.01339401 0.9333423  0.01435058 0.9994643
## [7]  {0.65}     => {United Kingdom} 0.01341898 0.9169182  0.01463487 0.9818768
## [8]  {0.39}     => {United Kingdom} 0.01457916 0.9501753  0.01534366 1.0174899
## [9]  {5}        => {United Kingdom} 0.02166514 0.9660814  0.02242579 1.0345228
## [10] {3.29}     => {United Kingdom} 0.02242963 0.9965011  0.02250838 1.0670976
## [11] {2.55}     => {United Kingdom} 0.02125792 0.8961134  0.02372235 0.9595980
## [12] {1.45}     => {United Kingdom} 0.02109465 0.8857166  0.02381648 0.9484647
## [13] {1.95}     => {United Kingdom} 0.02183417 0.8995015  0.02427363 0.9632261
## [14] {1.63}     => {United Kingdom} 0.02442346 1.0000000  0.02442346 1.0708444
## [15] {8}        => {United Kingdom} 0.02107544 0.8626464  0.02443114 0.9237601
## [16] {4.13}     => {United Kingdom} 0.02945798 1.0000000  0.02945798 1.0708444
## [17] {2.08}     => {United Kingdom} 0.02947334 0.9306162  0.03167078 0.9965452
## [18] {0.83}     => {United Kingdom} 0.03125396 0.9713450  0.03217596 1.0401594
## [19] {2.46}     => {United Kingdom} 0.03282905 1.0000000  0.03282905 1.0708444
## [20] {2.1}      => {United Kingdom} 0.03082177 0.9385821  0.03283865 1.0050754
## [21] {3.75}     => {United Kingdom} 0.03096968 0.9180617  0.03373376 0.9831013
## [22] {4.95}     => {United Kingdom} 0.03161508 0.9178563  0.03444447 0.9828813
## [23] {10}       => {United Kingdom} 0.03737952 0.8940960  0.04180705 0.9574377
## [24] {24}       => {United Kingdom} 0.03812864 0.8637946  0.04414087 0.9249896
## [25] {0.42}     => {United Kingdom} 0.04208557 0.9187353  0.04580815 0.9838226
## [26] {2.95}     => {United Kingdom} 0.04679162 0.9239172  0.05064483 0.9893715
## [27] {0.85}     => {United Kingdom} 0.04758685 0.9032046  0.05268668 0.9671916
## [28] {3}        => {United Kingdom} 0.06768266 0.9618912  0.07036415 1.0300359
## [29] {1.65}     => {United Kingdom} 0.06405804 0.9080982  0.07054087 0.9724319
## [30] {4}        => {United Kingdom} 0.06695274 0.9306598  0.07194116 0.9965918
## [31] {6}        => {United Kingdom} 0.06757317 0.8919399  0.07575979 0.9551288
## [32] {1.25}     => {United Kingdom} 0.08691794 0.9276723  0.09369466 0.9933927
## [33] {12}       => {United Kingdom} 0.09685251 0.8634052  0.11217504 0.9245726
## [34] {2}        => {United Kingdom} 0.15061870 0.9707342  0.15515956 1.0395053
## [35] {1}        => {United Kingdom} 0.28068443 0.9885267  0.28394218 1.0585583
## [36] {1, 3.29}  => {United Kingdom} 0.01252579 0.9998467  0.01252771 1.0706802
## [37] {1, 1.63}  => {United Kingdom} 0.01201484 1.0000000  0.01201484 1.0708444
## [38] {1, 4.13}  => {United Kingdom} 0.01692259 1.0000000  0.01692259 1.0708444
## [39] {0.83, 1}  => {United Kingdom} 0.01195338 0.9974355  0.01198411 1.0680982
## [40] {1, 2.46}  => {United Kingdom} 0.01780233 1.0000000  0.01780233 1.0708444
## [41] {1.65, 10} => {United Kingdom} 0.01084505 0.8803992  0.01231834 0.9427705
## [42] {2.95, 6}  => {United Kingdom} 0.01448120 0.8573866  0.01688993 0.9181276
## [43] {1, 2.95}  => {United Kingdom} 0.01054924 0.9879475  0.01067794 1.0579380
## [44] {0.85, 12} => {United Kingdom} 0.01160571 0.8474053  0.01369558 0.9074393
## [45] {1.65, 12} => {United Kingdom} 0.01146356 0.8202309  0.01397602 0.8783397
## [46] {1, 1.65}  => {United Kingdom} 0.01144243 0.9846281  0.01162107 1.0543835
## [47] {1.25, 12} => {United Kingdom} 0.02278499 0.8801662  0.02588714 0.9425211
## [48] {1.25, 2}  => {United Kingdom} 0.01128877 0.9860738  0.01144820 1.0559316
## [49] {1, 1.25}  => {United Kingdom} 0.01807317 0.9865786  0.01831904 1.0564722
##      count 
## [1]  486164
## [2]    6195
## [3]    6844
## [4]    6456
## [5]    6704
## [6]    6973
## [7]    6986
## [8]    7590
## [9]   11279
## [10]  11677
## [11]  11067
## [12]  10982
## [13]  11367
## [14]  12715
## [15]  10972
## [16]  15336
## [17]  15344
## [18]  16271
## [19]  17091
## [20]  16046
## [21]  16123
## [22]  16459
## [23]  19460
## [24]  19850
## [25]  21910
## [26]  24360
## [27]  24774
## [28]  35236
## [29]  33349
## [30]  34856
## [31]  35179
## [32]  45250
## [33]  50422
## [34]  78413
## [35] 146126
## [36]   6521
## [37]   6255
## [38]   8810
## [39]   6223
## [40]   9268
## [41]   5646
## [42]   7539
## [43]   5492
## [44]   6042
## [45]   5968
## [46]   5957
## [47]  11862
## [48]   5877
## [49]   9409
```

``` r
# To view the top 10 rules
inspect(association_rules[1:10])
```

```         
##      lhs       rhs              support    confidence coverage   lift     
## [1]  {}     => {United Kingdom} 0.93384248 0.9338425  1.00000000 1.0000000
## [2]  {7.95} => {United Kingdom} 0.01189959 0.9251792  0.01286193 0.9907230
## [3]  {5.79} => {United Kingdom} 0.01314622 1.0000000  0.01314622 1.0708444
## [4]  {4.25} => {United Kingdom} 0.01240093 0.9247959  0.01340937 0.9903125
## [5]  {0.55} => {United Kingdom} 0.01287730 0.9131027  0.01410280 0.9777909
## [6]  {5.95} => {United Kingdom} 0.01339401 0.9333423  0.01435058 0.9994643
## [7]  {0.65} => {United Kingdom} 0.01341898 0.9169182  0.01463487 0.9818768
## [8]  {0.39} => {United Kingdom} 0.01457916 0.9501753  0.01534366 1.0174899
## [9]  {5}    => {United Kingdom} 0.02166514 0.9660814  0.02242579 1.0345228
## [10] {3.29} => {United Kingdom} 0.02242963 0.9965011  0.02250838 1.0670976
##      count 
## [1]  486164
## [2]    6195
## [3]    6844
## [4]    6456
## [5]    6704
## [6]    6973
## [7]    6986
## [8]    7590
## [9]   11279
## [10]  11677
```

``` r
plot(association_rules)
```

![](lab7C-submission-markdown_files/figure-gfm/association%20rules%20creation-1.png)<!-- -->

``` r
### Remove redundant rules ----
subset_rules <-
  which(colSums(is.subset(association_rules,
                          association_rules)) > 1)
length(subset_rules)
```

```         
## [1] 48
```

``` r
association_rules_no_reps <- association_rules[-subset_rules]

summary(association_rules_no_reps)
```

```         
## set of 1 rules
## 
## rule length distribution (lhs + rhs):sizes
## 1 
## 1 
## 
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##       1       1       1       1       1       1 
## 
## summary of quality measures:
##     support         confidence        coverage      lift       count       
##  Min.   :0.9338   Min.   :0.9338   Min.   :1   Min.   :1   Min.   :486164  
##  1st Qu.:0.9338   1st Qu.:0.9338   1st Qu.:1   1st Qu.:1   1st Qu.:486164  
##  Median :0.9338   Median :0.9338   Median :1   Median :1   Median :486164  
##  Mean   :0.9338   Mean   :0.9338   Mean   :1   Mean   :1   Mean   :486164  
##  3rd Qu.:0.9338   3rd Qu.:0.9338   3rd Qu.:1   3rd Qu.:1   3rd Qu.:486164  
##  Max.   :0.9338   Max.   :0.9338   Max.   :1   Max.   :1   Max.   :486164  
## 
## mining info:
##  data ntransactions support confidence
##    tr        520606    0.01        0.8
##                                                                                 call
##  apriori(data = tr, parameter = list(support = 0.01, confidence = 0.8, maxlen = 10))
```

``` r
inspect(association_rules_no_reps)
```

```         
##     lhs    rhs              support   confidence coverage lift count 
## [1] {}  => {United Kingdom} 0.9338425 0.9338425  1        1    486164
```

``` r
write(association_rules_no_reps,
      file = "rules/association_rules_based_on_itemslist.csv")
```

## Finding Specific Rules {#finding-specific-rules}

This code chunk specifically focuses on mining association rules where
the consequent (right-hand side) of the rules is "DOORMAT NEW ENGLAND
AND SPACEBOY LUNCH BOX." It applies a minimum support threshold of 0.01
and a minimum confidence threshold of 0.8. The output is then inspected
to view the initial set of rules that match these criteria.

``` r
strawberry_charlotte_bag_association_rules <- # nolint
  apriori(tr, parameter = list(supp = 0.01, conf = 0.8),
          appearance = list(lhs = c("STRAWBERRY CHARLOTTE BAG", "WOODLAND CHARLOTTE BAG"), # nolint
                            default = "rhs"))
```

```         
## Apriori
## 
## Parameter specification:
##  confidence minval smax arem  aval originalSupport maxtime support minlen
##         0.8    0.1    1 none FALSE            TRUE       5    0.01      1
##  maxlen target  ext
##      10  rules TRUE
## 
## Algorithmic control:
##  filter tree heap memopt load sort verbose
##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
## 
## Absolute minimum support count: 5206 
## 
## set item appearances ...[2 item(s)] done [0.00s].
## set transactions ...[27319 item(s), 520606 transaction(s)] done [1.38s].
## sorting and recoding items ... [42 item(s)] done [0.02s].
## creating transaction tree ... done [0.39s].
## checking subsets of size 1 done [0.00s].
## writing ... [1 rule(s)] done [0.00s].
## creating S4 object  ... done [0.06s].
```

``` r
inspect(head(strawberry_charlotte_bag_association_rules))
```

```         
##     lhs    rhs              support   confidence coverage lift count 
## [1] {}  => {United Kingdom} 0.9338425 0.9338425  1        1    486164
```

## Visualizing the Rules {#visualizing-the-rules}

This code chunk filters association rules based on a confidence
threshold, generates various types of plots to visualize these rules,
and provides an interactive graph-based visualization of the top 10
rules with the highest confidence.

``` r
rules_to_plot <-
  association_rules_no_reps[quality(association_rules_no_reps)$confidence > 0.85]

#Plot SubRules
plot(rules_to_plot)
```

![](lab7C-submission-markdown_files/figure-gfm/visualizing-1.png)<!-- -->

``` r
plot(rules_to_plot, method = "two-key plot")
```

![](lab7C-submission-markdown_files/figure-gfm/visualizing-2.png)<!-- -->

``` r
top_10_rules_to_plot <- head(rules_to_plot, n = 10, by = "confidence")
plot(top_10_rules_to_plot, method = "graph",  engine = "htmlwidget")
```

```         
## PhantomJS not found. You can install it with webshot::install_phantomjs(). If it is installed, please make sure the phantomjs executable can be found via the PATH variable.
```

::: {#htmlwidget-530b8073dddc28531b6c .visNetwork .html-widget .html-fill-item-overflow-hidden .html-fill-item style="width:576px;height:384px;"}
:::

```{=html}
<script type="application/json" data-for="htmlwidget-530b8073dddc28531b6c">{"x":{"nodes":{"id":[1,2],"label":["United Kingdom","rule 1"],"group":[1,2],"value":[1,50.5],"color":["#CBD2FC","#EE9797"],"title":["United Kingdom","<B>[1]<\/B><BR><B>{}<\/B><BR>&nbsp;&nbsp; => <B>{United Kingdom}<\/B><BR><BR>support = 0.934<BR>confidence = 0.934<BR>coverage = 1<BR>lift = 1<BR>count = 486000<BR>order = 1<BR>id = 1"],"shape":["box","circle"],"x":[-1,1],"y":[-1,1]},"edges":{"from":[2],"to":[1],"arrows":["to"]},"nodesToDataframe":true,"edgesToDataframe":true,"options":{"width":"100%","height":"100%","nodes":{"shape":"dot","physics":false},"manipulation":{"enabled":false},"edges":{"smooth":false},"physics":{"stabilization":false},"interaction":{"hover":true,"zoomSpeed":1}},"groups":["1","2"],"width":null,"height":null,"idselection":{"enabled":true,"style":"width: 150px; height: 26px","useLabels":true,"main":"Select by id"},"byselection":{"enabled":false,"style":"width: 150px; height: 26px","multiple":false,"hideColor":"rgba(200,200,200,0.5)","highlight":false},"main":null,"submain":null,"footer":null,"background":"rgba(0, 0, 0, 0)","igraphlayout":{"type":"square"},"tooltipStay":300,"tooltipStyle":"position: fixed;visibility:hidden;padding: 5px;white-space: nowrap;font-family: verdana;font-size:14px;font-color:#000000;background-color: #f5f4ed;-moz-border-radius: 3px;-webkit-border-radius: 3px;border-radius: 3px;border: 1px solid #808074;box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);","highlight":{"enabled":true,"hoverNearest":true,"degree":1,"algorithm":"all","hideColor":"rgba(200,200,200,0.5)","labelOnly":true},"collapse":{"enabled":false,"fit":false,"resetHighlight":true,"clusterOptions":null,"keepCoord":true,"labelSuffix":"(cluster)"}},"evals":[],"jsHooks":[]}</script>
```
