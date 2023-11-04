# STEP 1. Install and Load the Required Packages ----
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

# STEP 2. Load and pre-process the dataset ----
itemslist <- read_excel("data/Assignment-1_Data.xlsx")
dim(itemslist)

### Handle missing values ----
# Are there missing values in the dataset?
any_na(itemslist)

# How many?
n_miss(itemslist)

# What is the proportion of missing data in the entire dataset?
prop_miss(itemslist)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(itemslist)

# Which variables contain the most missing values?
gg_miss_var(itemslist)

# Which combinations of variables are missing together?
gg_miss_upset(itemslist)

#### OPTION 1: Remove the observations with missing values ----
itemslist_removed_obs <- itemslist %>% dplyr::filter(complete.cases(.))

# We end up with 388,023 observations to create the association rules
# instead of the initial 522,064 observations.
dim(itemslist_removed_obs)

# Are there missing values in the dataset?
any_na(itemslist_removed_obs)

#### OPTION 2: Remove the variables with missing values ----
# The `CustomerID` variable will not be used to create association rules.
# We will use the `BillNo` instead.
itemslist_removed_vars <-
  itemslist %>% dplyr::select(-CustomerID)

dim(itemslist_removed_vars)

# Are there missing values in the dataset?
any_na(itemslist_removed_vars)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(itemslist_removed_vars)

# We now remove the observations that do not have a value for the description
# variable.
itemslist_removed_vars_obs <- itemslist_removed_vars %>% filter(complete.cases(.))

# We end up with 520,606 observations to create the association rules
# instead of the initial 522,064 observations.
# This is better than OPTION 1 which resulted in 388,023 observations to
# create the association rules.
dim(itemslist_removed_vars_obs)

## Identify categorical variables ----
# Ensure the country is recorded as categorical data
itemslist_removed_vars_obs %>% mutate(Country = as.factor(Country))

# Also ensure that the Itemname is recorded
# as categorical data
itemslist_removed_vars_obs %>% mutate(Itemname = as.factor(Itemname))
str(itemslist_removed_vars_obs)

dim(itemslist_removed_vars_obs)
head(itemslist_removed_vars_obs)

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
View(itemslist_removed_vars_obs)

# We can save the pre-processing progress made so far
write.csv(itemslist_removed_vars_obs,
          file = "data/itemslist_data_before_single_transaction_format.csv",
          row.names = FALSE)

itemslist_removed_vars_obs <-
  read.csv(file = "data/itemslist_data_before_single_transaction_format.csv")

## Create a transaction data frame using the "basket format" ----
str(itemslist_removed_vars_obs)

# ddply is used to split a data frame, apply a function to the split data,
# and then return the result back in a data frame.
### OPTION 1 ----
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

# STEP 2. Basic EDA ----
# Create an item frequency plot for the top 10 items
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

# STEP 3. Create the association rules ----
association_rules <- apriori(tr,
                             parameter = list(support = 0.01,
                                              confidence = 0.8,
                                              maxlen = 10))
#Print the association rules ----
summary(association_rules)
inspect(association_rules)
# To view the top 10 rules
inspect(association_rules[1:10])
plot(association_rules)

### Remove redundant rules ----
subset_rules <-
  which(colSums(is.subset(association_rules,
                          association_rules)) > 1)
length(subset_rules)
association_rules_no_reps <- association_rules[-subset_rules]

summary(association_rules_no_reps)
inspect(association_rules_no_reps)

write(association_rules_no_reps,
      file = "rules/association_rules_based_on_itemslist.csv")

# STEP 4. Find specific rules ----
doormat_new_england_and_spaceboy_lunch_box_association_rules <- # nolint
  apriori(tr, parameter = list(supp = 0.01, conf = 0.8),
          appearance = list(default = "lhs",
                            rhs = "DOORMAT NEW ENGLAND "))
inspect(head(doormat_new_england_and_spaceboy_lunch_box_association_rules))

# STEP 5. Visualize the rules ----
rules_to_plot <-
  association_rules_no_reps[quality(association_rules_no_reps)$confidence > 0.85]

#Plot SubRules
plot(rules_to_plot)
plot(rules_to_plot, method = "two-key plot")

top_10_rules_to_plot <- head(rules_to_plot, n = 10, by = "confidence")
plot(top_10_rules_to_plot, method = "graph",  engine = "htmlwidget")
