# Loan Status Prediction Project
Predicting loan defaults is important to minimize financial risk and ensure the stability of lending institutions by identifying borrowers who are more likely to default on their loan obligations. In this project, logistic regression, random forest, and XGBoost models were trained to determine loan status and prevent potential default.
## Load Library
```
library(tidyverse)
library(tidyr)
library(tidymodels)
library(tidytext)
library(janitor)
library(skimr)
library(vip)
library(parallel)
library(doParallel)
library(rpart.plot)
library(textrecipes)
library(stringi)
library(xgboost)
library(DALEXtra)
library(DALEX)
library(MASS)
library(solitude)
library(rpart.plot)
library(ggpubr)
```
## Import Data
```
loan <- read_csv("loan_train.csv", na = c("NA", "", "-")) %>% 
  mutate(int_rate = as.numeric(str_replace(int_rate, "%", "")),
         revol_util = as.numeric(str_replace(revol_util, "%", "")),
         empl_year = word(emp_length)) %>% clean_names()

skim(loan)
```
## Exploratory Analysis
```
loan <- loan %>%
  mutate_if(is.character, as.factor) %>%
  mutate(loan_status = factor(loan_status))

loan %>%
  count(loan_status) %>%
  mutate(pct = n/sum(n)) -> default_rate

default_rate %>%
  ggplot(aes(x = loan_status, y = pct)) +
  geom_col() +
  geom_text(aes(label = round(pct, 4)), color = "red") +
  labs(title = "Loan Default Rate")
```
### Explore Factors
```
char_explore <- function(col){loan %>%
    na.omit() %>%
    ggplot(., aes(!!as.name(col))) +
    geom_bar(aes(fill=loan_status), position="fill") +
    theme(axis.text.x=element_text(angle=90, hjust=1)) +
    labs(title = col)
}

for (column in names(loan %>% select_if(is.factor))){
  chrt <- char_explore(column)
  print(chrt)
}
```
### Explore Numeric
```
n_cols <- names(loan %>% select_if(is.numeric) %>% dplyr::select(-id, -member_id))
my_hist <- function(col){
  loan %>%
    summarise(n = n(),
              n_miss = sum(is.na(!!as.name(col))),
              n_dist = n_distinct(!!as.name(col)),
              mean = round(mean(!!as.name(col), na.rm = TRUE), 2),
              min = min(!!as.name(col), na.rm = TRUE),
              max = max(!!as.name(col), na.rm = TRUE)) -> col_summary
  
  p <- ggtexttable(col_summary, rows = NULL,
                   theme = ttheme("mOrange"))
  
  l1 <- loan %>%
    ggplot(aes(x = !!as.name(col), fill = loan_status)) +
    geom_histogram(bins = 30) +
    labs(title = col)
  
  plt <- ggarrange(l1, p, ncol = 1, nrow = 2, heights = c(1, 0.3))
  print(plt)
}

for (c in n_cols){
  my_hist(c)
}
```
### Correlation Matrix
```
cor <- loan %>%
  na.omit() %>%
  select_if(is.numeric) %>%
  cor() %>%
  as.data.frame() %>%
  rownames_to_column(var = "variable")

cor %>%
  pivot_longer(cols = c("loan_amnt",
                        "funded_amnt",
                        "funded_amnt_inv",
                        "installment",
                        "annual_inc",
                        "dti",
                        "delinq_2yrs",
                        "fico_range_low",
                        "fico_range_high",
                        "inq_last_6mths",
                        "mths_since_last_delinq",
                        "mths_since_last_record",
                        "open_acc",
                        "pub_rec",
                        "revol_bal",
                        "total_acc",
                        "out_prncp",
                        "out_prncp_inv",
                        "total_rec_late_fee",
                        "delinq_amnt"),
               names_to = "name",
               values_to = "correlation") %>%
  ggplot(aes(x = variable, y = name, fill = correlation)) +
  geom_tile() +
  labs(title = "Correlation Matrix",
       x = "Variable",
       y = "Variable") +
  scale_fill_gradient2(mid = "#FBFEF9",
                       low = "#0C6291",
                       high = "#A63446") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  geom_text(aes(label = round(correlation, 2)), color = "Black", size = 1.5)
```
## Data Partition
```
set.seed(123)

split <- initial_split(loan, prop = 0.7, strata = loan_status)
train <- training(split)
test <- testing(split)

sprintf("Train PCT : %1.2f%%", nrow(train)/nrow(loan) * 100)
sprintf("Test PCT  : %1.2f%%", nrow(test)/nrow(loan) * 100)

loan_fold <- train %>% sample_frac(0.2)
```
## Recipe
```
loan_recipe <- recipe(loan_status ~ ., data = train) %>%
  step_rm(emp_title, url, desc, title, zip_code, earliest_cr_line, last_pymnt_d, addr_state,
          issue_d, next_pymnt_d, last_credit_pull_d, id, member_id) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  themis::step_downsample(loan_status, under_ratio = 3) %>%
  step_nzv(all_nominal_predictors())
```
## Model Workflows
### Logistic Model
```
log_model <- logistic_reg() %>%
  set_mode("classification") %>%
  set_engine("glm")

log_workflow_fit <- workflow() %>%
  add_recipe(loan_recipe) %>%
  add_model(log_model) %>%
  fit(train)
```
### XGBoost Model
```
xgb_model <- boost_tree(trees = 20) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_workflow_fit <- workflow() %>%
  add_recipe(loan_recipe) %>%
  add_model(xgb_model) %>%
  fit(train)
```
### Random Forest Model
```
rf_model <- rand_forest(trees = 100) %>%
  set_engine("ranger", num.thread = 8, importance = "permutation") %>%
  set_mode("classification")

rf_workflow_fit <- workflow() %>%
  add_recipe(loan_recipe) %>%
  add_model(rf_model) %>%
  fit(train)
```
## Model Evaluation
```
evaluate_models <- function(model_workflow, model_name){
  # -- make predictions
  score_train <- bind_cols(
    predict(model_workflow, train, type = "prob"),
    predict(model_workflow, train, type = "class"),
    train) %>%
    mutate(part = "train")
  
  score_test <- bind_cols(
    predict(model_workflow, test, type = "prob"),
    predict(model_workflow, test, type = "class"),
    test) %>%
    mutate(part = "test")
  
  options(yardstick.event_first = FALSE)
  
  bind_rows(score_train, score_test) %>%
    group_by(part) %>%
    metrics(loan_status, .pred_default, estimate = .pred_class) %>%
    pivot_wider(id_cols = part, names_from = .metric, values_from = .estimate) %>%
    mutate(model_name = model_name) %>% print()
  
  # -- precision
  bind_rows(score_train, score_test) %>%
    group_by(part) %>%
    precision(loan_status, .pred_class) %>%
    mutate(model_name = model_name) %>% print()

# -- recall
  bind_rows(score_train, score_test) %>%
    group_by(part) %>%
    recall(loan_status, .pred_class) %>%
    mutate(model_name = model_name) %>% print()
  
  # -- roc curve
  bind_rows(score_train, score_test) %>%
    group_by(part) %>%
    roc_curve(truth = loan_status, predicted = .pred_default) %>%
    autoplot() +
    geom_vline(xintercept = 0.04, color = "red", linetype = "longdash") +
    geom_vline(xintercept = 0.20, color = "black", linetype = "longdash") +
    labs(title = model_name, x = "FPR(1 - specificity)", y = "TPR(recall)") -> roc_chart
  
  print(roc_chart)
  
  # -- operating range 0 - 10%
  operating_range <- score_test %>%
    roc_curve(loan_status, .pred_default) %>%
    mutate(fpr = 1 - round(specificity, 2),
           tpr = round(sensitivity, 3),
           threshold = 1 - round(.threshold, 3)) %>%
    #dplyr::select(fpr, tpr, threshold) %>%
    group_by(fpr) %>%
    summarise(threshold = round(mean(threshold), 3),
              tpr = mean(tpr)) %>%
    #mutate(precision = tpr/(fpr+tpr)) %>%
    #mutate_if(is.numeric, round, 2) %>%
    arrange(fpr) %>%
    filter(fpr <= 0.1)
  
  print(operating_range)
  
  # -- score distribution
  score_test %>%
    ggplot(aes(.pred_default, fill = loan_status)) +
    geom_histogram(bins = 50) +
    geom_vline(aes(xintercept = .5, color = "red")) +
    geom_vline(aes(xintercept = .3, color = "green")) +
    geom_vline(aes(xintercept = .7, color = "blue")) +
    labs(title = model_name) -> score_dist
  
  print(score_dist)
  
  # -- variable importance
  model_workflow %>%
    extract_fit_parsnip() %>%
    vip(30) +
    labs(model_name) -> vip_model
  
  print(vip_model)
}

evaluate_models(xgb_workflow_fit, "XGBoost Model")
evaluate_models(rf_workflow_fit, "Random Forest Model")
evaluate_models(log_workflow_fit, "Logistic Model")
```
![Picture4](https://github.com/dingy21/loan-status/assets/134649288/de542a5e-f335-40fc-a4d7-b34cca49ff04)
## Top Predictions
```
top_tp <- xgb_scored_test %>%
  filter(.pred_class == loan_status) %>%
  filter(loan_status == "default") %>%
  slice_max(order_by = .pred_default, n = 10)

top_tn <- xgb_scored_test %>%
  filter(.pred_class == loan_status) %>%
  filter(loan_status == "current") %>%
  slice_max(order_by = .pred_default, n = 10)

top_fp <- xgb_scored_test %>%
  filter(.pred_class != loan_status) %>%
  filter(loan_status == "current") %>%
  slice_max(order_by = .pred_default, n = 10)

bottom_fn <- xgb_scored_test %>%
  filter(.pred_class != loan_status) %>%
  filter(loan_status == "default") %>%
  slice_min(order_by = .pred_default, n = 10)
```
## Partial Dependence Plot
```
grid <- recipe(loan_status ~ ., data = train) %>%
  step_profile(all_predictors(), -last_pymnt_amnt, profile = vars(last_pymnt_amnt)) %>%
  prep() %>% juice()

predict(xgb_workflow_fit, grid, type = "prob") %>%
  bind_cols(grid %>% dplyr::select(last_pymnt_amnt)) %>%
  ggplot(aes(y = .pred_default, x = last_pymnt_amnt)) +
  geom_path() + stat_smooth() +
  labs(title = "Partial Dependence Plot - Last Payment Amount")
```
```
# -- create explainer object
xgb_explainer <- explain_tidymodels(xgb_workflow_fit, data = train,
                                    y = train$loan_status, verbose = TRUE)

# -- profile the variable of interest
pdp_last_pymnt_amnt <- model_profile(xgb_explainer, variables = "last_pymnt_amnt")

# -- plot it
plot(pdp_last_pymnt_amnt) + labs(title = "Partial Dependence Plot",
                                 subtitle = " ",
                                 x = "last_pymnt_amnt",
                                 y = "Average Impact on Prediction")

# -- PDP: late fees received to date
pdp_total_rec_late_fee <- model_profile(xgb_explainer, variables = "total_rec_late_fee")
plot(pdp_total_rec_late_fee) + labs(title = "Partial Dependence Plot",
                                    subtitle = " ",
                                    x = "Late Fees Received to Date",
                                    y = "Average Impact on Prediction")

# -- PDP: interest rate on the loan
pdp_int_rate <- model_profile(xgb_explainer, variables = "int_rate")
plot(pdp_int_rate) + labs(title = "Partial Dependence Plot",
                          subtitle = " ",
                          x = "Interest Rate on the Loan",
                          y = "Average Impact on Prediction")

# -- PDP: monthly payment owed by the borrower if the loan originates
pdp_installment <- model_profile(xgb_explainer, variables = "installment")
plot(pdp_installment) + labs(title = "Partial Dependence Plot",
                             subtitle = " ",
                             x = "Monthly Payment Owed by Borrower (if the loan originates)",
                             y = "Average Impact on Prediction")

# -- PDP: remaining outstanding principal for total amount funded
pdp_out_prncp <- model_profile(xgb_explainer, variables = "out_prncp")
plot(pdp_out_prncp) + labs(title = "Partial Dependence Plot",
                           subtitle = " ",
                           x = "Remaining Outstanding Principal",
                           y = "Average Impact on Prediction")

# -- PDP: total amount committed by investors for that loan at that point in time
pdp_funded_amnt_inv <- model_profile(xgb_explainer, variables = "funded_amnt_inv")
plot(pdp_funded_amnt_inv) + labs(title = "Partial Dependence Plot",
                                 subtitle = " ",
                                 x = "Total Amount Committed by Investors",
                                 y = "Average Impact on Prediction")

# -- PDP: self-reported annual income provided by the borrower during registration
pdp_annual_inc <- model_profile(xgb_explainer, variables = "annual_inc")
plot(pdp_annual_inc) + labs(title = "Partial Dependence Plot",
                            subtitle = " ",
                            x = "Borrower's Annual Income",
                            y = "Average Impact on Prediction")

# -- PDP: listed amount of the loan applied for by the borrower
pdp_loan_amnt <- model_profile(xgb_explainer, variables = "loan_amnt")
plot(pdp_loan_amnt) + labs(title = "Partial Dependence Plot",
                           subtitle = " ",
                           x = "Amount of Loan",
                           y = "Average Impact on Prediction")
```
![Picture5](https://github.com/dingy21/loan-status/assets/134649288/80b376a4-6c79-4fc3-879e-d2cd7e727886)![Picture6](https://github.com/dingy21/loan-status/assets/134649288/5b0644c6-057e-4268-8e80-a77ec9a15f81)
![Picture7](https://github.com/dingy21/loan-status/assets/134649288/d15d5e2e-c8c2-4b50-bc23-bc3efa8b1aba)![Picture8](https://github.com/dingy21/loan-status/assets/134649288/ea8598e4-5165-4258-9f52-1cf1b37be503)
![Picture9](https://github.com/dingy21/loan-status/assets/134649288/b53e6748-742b-40b8-98a8-1bcf0fa2a494)![Picture10](https://github.com/dingy21/loan-status/assets/134649288/7d6f59e3-2b6c-4166-acba-a1d9aed55aa5)
## Shap & Breakdown Plots
```
tidy_explainer <- explain_tidymodels(xgb_workflow_fit, data = test,
                                     y = test$loan_status, label = "tidymodels")

shap_explain <- predict_parts(tidy_explainer, top_tp %>% head(1), type = "shap")
plot(shap_explain)

score <- 0.67

as_tibble(shap_explain) %>%
  group_by(variable) %>%
  summarise(contribution = sum(contribution)) %>%
  top_n(wt = abs(contribution), 10) %>%
  mutate(pos_neg = if_else(contribution < 0, "neg", "pos")) %>%
  arrange(desc(contribution)) %>%
  ggplot(aes(x = contribution, y = reorder(variable, contribution), fill = pos_neg)) +
  geom_col() +
  labs(title = paste("Shap Explainer, Predicted Score:", score))


breakdown_explain <- predict_parts(tidy_explainer, top_tp %>% head(1),
                                   type = "break_down_interactions")
plot(breakdown_explain) + labs(title = paste("Breakdown Plot, Predicted Score:", score))
```
### True Positive
```
breakdown_explainer <- function(row){
  breakdown_explain <- predict_parts(tidy_explainer, row, type = "break_down_interactions")
  plot(breakdown_explain) +
  labs(title = paste("Breakdown Plot, Predicted Score:", round(row$.pred_default, 3)))
}

for (row in 1:2){
  dat <- top_tp[row,]
  print(breakdown_explainer(dat))
}

for (row in 3:3){
  dat <- top_tp[row,]
  print(breakdown_explainer(dat))
}
```
![Picture12](https://github.com/dingy21/loan-status/assets/134649288/df6f36c6-efc9-42ff-9f24-d5283d27465e)![Picture13](https://github.com/dingy21/loan-status/assets/134649288/be89874e-3128-49b5-8dc9-6ea81c2d72cf)
### False Positive
```
shap_explain <- predict_parts(tidy_explainer, top_fp %>% head(1), type = "shap")
plot(shap_explain)

as_tibble(shap_explain) %>%
  group_by(variable) %>%
  summarise(contribution = sum(contribution)) %>%
  top_n(wt = abs(contribution), 10) %>%
  mutate(pos_neg = if_else(contribution < 0, "neg", "pos")) %>%
  arrange(desc(contribution)) %>%
  ggplot(aes(x = contribution, y = reorder(variable, contribution), fill = pos_neg)) +
  geom_col() +
  labs(title = paste("Shap Explainer, Predicted Score:", score))

for (row in 1:3){
  dat <- top_fp[row,]
  print(breakdown_explainer(dat))
}
```
![Picture14](https://github.com/dingy21/loan-status/assets/134649288/50a3e626-ed27-46f2-a5e7-db21e65f1773)![Picture15](https://github.com/dingy21/loan-status/assets/134649288/cba21aaa-0337-402e-9f69-9827c82f465a)
### True Negative
```
shap_explain <- predict_parts(tidy_explainer, top_tn %>% head(1), type = "shap")
plot(shap_explain)

as_tibble(shap_explain) %>%
  group_by(variable) %>%
  summarise(contribution = sum(contribution)) %>%
  top_n(wt = abs(contribution), 10) %>%
  mutate(pos_neg = if_else(contribution < 0, "neg", "pos")) %>%
  arrange(desc(contribution)) %>%
  ggplot(aes(x = contribution, y = reorder(variable, contribution), fill = pos_neg)) +
  geom_col() +
  labs(title = paste("Shap Explainer, Predicted Score:", score))

for (row in 1:3){
  dat <- top_tn[row,]
  print(breakdown_explainer(dat))
}
```
![Picture16](https://github.com/dingy21/loan-status/assets/134649288/fd53f6a8-59ef-4654-b0d8-18c41e4bf10b)![Picture17](https://github.com/dingy21/loan-status/assets/134649288/369869e0-0bb2-4aca-8ff9-18672b4492ff)
### False Negative
```
shap_explain <- predict_parts(tidy_explainer, bottom_fn %>% head(1), type = "shap")
plot(shap_explain)

as_tibble(shap_explain) %>%
  group_by(variable) %>%
  summarise(contribution = sum(contribution)) %>%
  top_n(wt = abs(contribution), 10) %>%
  mutate(pos_neg = if_else(contribution < 0, "neg", "pos")) %>%
  arrange(desc(contribution)) %>%
  ggplot(aes(x = contribution, y = reorder(variable, contribution), fill = pos_neg)) +
  geom_col() +
  labs(title = paste("Shap Explainer, Predicted Score:", score))

for (row in 1:3){
  dat <- bottom_fn[row,]
  print(breakdown_explainer(dat))
}
```
![Picture25](https://github.com/dingy21/loan-status/assets/134649288/03648c4e-c202-487b-8ec0-97a4739b3cea)
![Picture26](https://github.com/dingy21/loan-status/assets/134649288/c539d391-c4a3-47ad-a640-615be0200657)
## Isolation Recipe
```
iso_recipe <- recipe(~.,loan) %>%
  step_rm(id,member_id,emp_title,issue_d,url,desc,title,zip_code,earliest_cr_line,last_pymnt_d,
          next_pymnt_d,last_credit_pull_d) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

iso_prep <- bake(iso_recipe %>% prep(), loan)

# -- deal w. numeric
iso_numeric <- loan %>% select_if(is.numeric)

iso_recipe <- recipe(~.,iso_numeric) %>%
  step_rm(id, member_id) %>%
  step_impute_median(all_numeric()) %>%
  prep()
```
## Isolation Forest
```
iso_forest <- isolationForest$new(sample_size = 256,
                                  num_trees = 100,
                                  max_depth = ceiling(log2(256)))

iso_forest$fit(iso_prep)
```
## Predict Training
```
pred_train <- iso_forest$predict(iso_prep)

pred_train %>%
  summarise(n = n(),
            min = min(average_depth),
            max = max(average_depth),
            mean = mean(average_depth),
            min_score = min(anomaly_score),
            max_score = max(anomaly_score),
            mean_score = mean(anomaly_score))

# -- average tree depth
pred_train %>%
  ggplot(aes(average_depth)) +
  geom_histogram(bins=20) +
  geom_vline(xintercept = 7.4, linetype = "dotted",
             color = "blue", size = 1.5) + 
  labs(title = "Isolation Forest Tree Depth")

# -- average anomaly score
pred_train %>%
  ggplot(aes(anomaly_score)) +
  geom_histogram(bins=20) +
  geom_vline(xintercept = 0.607, linetype = "dotted",
             color = "blue", size = 1.5) +
  labs(title = "Isolation Forest Anomaly Score")
```
## Create a Data Frame with Anomaly Flag
```
train_pred <- bind_cols(iso_forest$predict(iso_prep), iso_prep) %>%
  mutate(anomaly = as.factor(if_else(average_depth <= 7.4, "Anomaly", "Normal")))

train_pred %>%
  arrange(average_depth) %>%
  count(anomaly)

head(train_pred, 10)
```
![Picture24](https://github.com/dingy21/loan-status/assets/134649288/9a6425dc-22cd-43b2-adcb-db3d5a87f789)
## Fit a Tree
```
fmla <- as.formula(paste("anomaly ~ ", paste(iso_prep %>% colnames(), collapse= "+")))

outlier_tree <- decision_tree(min_n = 2, tree_depth = 3,
                              cost_complexity = .01) %>%
  set_mode("classification") %>%
  set_engine("rpart") %>%
  fit(fmla, data = train_pred)

outlier_tree$fit

# -- plotting decision trees
library(rpart.plot)
rpart.plot(outlier_tree$fit, clip.right.labs = FALSE,
           branch = .3, under = TRUE, roundint = FALSE, extra = 3)
```
![Picture22](https://github.com/dingy21/loan-status/assets/134649288/c9e8e3eb-ad3d-4d39-822a-5c4ce53627b6)
## Global Anomaly Rules
```
anomaly_rules <- rpart.rules(outlier_tree$fit, roundint = FALSE,
                             extra = 4, cover = TRUE, clip.facs = TRUE) %>%
  clean_names() %>%
  mutate(rule = "IF")

rule_cols <- anomaly_rules %>% dplyr::select(starts_with("x_")) %>% colnames()

for (col in rule_cols){anomaly_rules <- anomaly_rules %>%
  mutate(rule = paste(rule, !!as.name(col)))
}

anomaly_rules %>%
  as.data.frame() %>%
  filter(anomaly == "Anomaly") %>%
  mutate(rule = paste(rule, " THEN ", anomaly)) %>%
  mutate(rule = paste(rule, " coverage ", cover)) %>%
  dplyr::select(rule)

anomaly_rules %>%
  as.data.frame() %>%
  filter(anomaly == "Normal") %>%
  mutate(rule = paste(rule, " THEN ", anomaly)) %>%
  mutate(rule = paste(rule, " coverage ", cover)) %>%
  dplyr::select(rule)
```
```
pred_train <- bind_cols(iso_forest$predict(iso_prep), iso_prep)

pred_train %>%
  arrange(desc(anomaly_score)) %>%
  filter(average_depth <= 7.4)
  ```
## Local Anomaly Rules
```
fmla <- as.formula(paste("anomaly ~ ", paste(iso_prep %>% colnames(), collapse= "+")))

# -- identify observation as anomaly
pred_train %>%
  mutate(anomaly = as.factor(if_else(id==28006, "Anomaly", "Normal"))) -> local_df

local_tree <-  decision_tree(mode = "classification", tree_depth = 4,
                             min_n = 1, cost_complexity = 0.01) %>%
  set_engine("rpart") %>%
  fit(fmla, local_df)

local_tree$fit

rpart.rules(local_tree$fit, extra = 4, cover = TRUE, clip.facs = TRUE, roundint = FALSE)
rpart.plot(local_tree$fit, roundint = FALSE, extra = 3)

anomaly_rules <- rpart.rules(local_tree$fit, extra = 4, cover = TRUE, clip.facs = TRUE) %>%
  clean_names() %>%
  filter(anomaly == "Anomaly") %>%
  mutate(rule = "IF") 

rule_cols <- anomaly_rules %>% dplyr::select(starts_with("x_")) %>% colnames()

for (col in rule_cols){
  anomaly_rules <- anomaly_rules %>%
    mutate(rule = paste(rule, !!as.name(col)))
}

as.data.frame(anomaly_rules) %>% dplyr::select(rule, cover)
```
![Picture23](https://github.com/dingy21/loan-status/assets/134649288/3d388674-3052-4d48-91ba-3d07b307b815)
```
local_explainer <- function(ID){
  fmla <- as.formula(paste("anomaly ~ ", paste(iso_prep %>% colnames(), collapse= "+")))
  
  pred_train %>%
    mutate(anomaly = as.factor(if_else(id==ID, "Anomaly", "Normal"))) -> local_df
  
  local_tree <- decision_tree(mode="classification",
                              tree_depth = 3,
                              min_n = 1,
                              cost_complexity = 0) %>%
    set_engine("rpart") %>%
    fit(fmla, local_df)
  
  local_tree$fit

  rpart.plot(local_tree$fit, roundint = FALSE, extra = 3) %>% print()
  
  anomaly_rules <- rpart.rules(local_tree$fit, extra = 4, cover = TRUE, clip.facs = TRUE) %>%
    clean_names() %>%
    filter(anomaly == "Anomaly") %>%
    mutate(rule = "IF")
  
  
  rule_cols <- anomaly_rules %>% dplyr::select(starts_with("x_")) %>% colnames()
  
  for (col in rule_cols){
    anomaly_rules <- anomaly_rules %>%
      mutate(rule = paste(rule, !!as.name(col)))
    }
  
  as.data.frame(anomaly_rules) %>%
    dplyr::select(rule, cover) %>%
    print()
  }

pred_train %>%
  slice_max(order_by = anomaly_score, n = 5) %>%
  pull(id) -> anomaly_vect

for (anomaly_id in anomaly_vect){
  print(anomaly_id)
  local_explainer(anomaly_id)
  }
  
```
