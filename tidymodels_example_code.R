# Environment Setup
source(".Rprofile")
options(tidymodels.dark = TRUE)

# Step 0: Load Libraries & Data
library(tidymodels)
library(titanic)
library(vip)  # for variable importance plots

# Load and prepare the Titanic training data
titanic_data <- titanic::titanic_train |> 
  as_tibble() |>
  mutate(Survived = factor(Survived, levels = c(0, 1)))  # Ensure target is a factor

glimpse(titanic_data)

# Step 1: Split the Data
set.seed(123)
titanic_split <- initial_split(
  data = titanic_data, 
  prop = 0.75, 
  strata = Survived
)

titanic_split

train_data <- training(titanic_split)
test_data  <- testing(titanic_split)

# Step 2: Create a Modeling Recipe
titanic_recipe <- recipe(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, 
                         data = train_data) |>
  step_impute_median(all_numeric_predictors()) |>      # Impute missing numeric values
  step_dummy(all_nominal_predictors()) |>              # Convert categorical predictors to dummy variables
  step_normalize(all_numeric_predictors())             # Normalize numeric predictors

titanic_recipe

# Step 3: Specify the Model (Logistic Regression with glm)
model_spec <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

model_spec

# Step 4: Create a Workflow
titanic_workflow <- workflow() |>
  add_recipe(titanic_recipe) |>
  add_model(model_spec)

titanic_workflow

# Step 5: Train the Model using last_fit() (training on the training set and evaluating on the test set)
titanic_fit <- last_fit(titanic_workflow,
                        split = titanic_split, 
                        metrics = metric_set(roc_auc, accuracy))

titanic_fit

# Step 6: Evaluate the Model
metrics <- collect_metrics(titanic_fit)
metrics

# Step 7: Visualize Model Performance (Confusion Matrix)
predictions <- titanic_fit |> collect_predictions()
conf_matrix <- predictions |> 
  conf_mat(truth = Survived, 
           estimate = .pred_class)

autoplot(conf_matrix, type = "heatmap")

# --- Now, let's move into hyperparameter tuning using glmnet ---
# Step 8: Specify a Tunable Model using glmnet
model_spec_tune <- logistic_reg(
  penalty = tune(), 
  mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("classification")

# Create a tuning workflow with the same recipe
tune_workflow <- workflow() |>
  add_recipe(titanic_recipe) |>
  add_model(model_spec_tune)

# Step 9: Create Cross-Validation Folds
set.seed(234)
titanic_folds <- vfold_cv(train_data, v = 5, strata = Survived)
titanic_folds

# Step 10: Define the Tuning Grid (using Latin Hypercube sampling)
log_grid <- grid_space_filling(
  penalty(), 
  mixture(), 
  size = 10
)

log_grid

# Step 11: Tune the Model
set.seed(345)
log_tuning_results <- tune_grid(
  tune_workflow,
  resamples = titanic_folds,
  grid = log_grid,
  metrics = metric_set(roc_auc, accuracy)
)

log_tuning_results

# Step 12: Visualize Tuning Results
best_models <- show_best(log_tuning_results, metric = "roc_auc")
best_models

autoplot(log_tuning_results)

# Step 13: Select the Best Model and Finalize Workflow
best_params <- select_best(log_tuning_results, metric = "roc_auc")
final_workflow <- finalize_workflow(tune_workflow, best_params)

# Step 14: Final Fit on the Test Set
final_fit <- last_fit(final_workflow, titanic_split)
final_metrics <- collect_metrics(final_fit)
final_metrics

# Step 15: Variable Importance
# Extract the fitted workflow and model, then display variable importance
fitted_workflow <- extract_workflow(final_fit)
fitted_model <- extract_fit_parsnip(fitted_workflow)

vip(fitted_model)

# Extra! ------------

# Let's work with WORKFLOWSETS
### https://workflowsets.tidymodels.org/

library(tictoc)
library(workflowsets)
library(finetune)
library(embed)
library(bonsai)
library(future)
library(xgboost)
library(ranger)

glm_spec <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

glmnet_spec <- logistic_reg(
  penalty = tune(), 
  mixture = tune()) |>
  set_engine("glmnet") |>
  set_mode("classification")

rpart_spec <- decision_tree(cost_complexity = tune(), 
                            tree_depth = tune(), 
                            min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("classification") 

rf_spec <- rand_forest(mtry = tune(), 
                       trees = tune(), 
                       min_n = tune()) |> 
  set_engine("ranger") |> 
  set_mode("classification") 

xgb_spec <- boost_tree(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

nn_spec <- mlp(epochs = 1000,
               hidden_units = tune(),
               penalty = tune()) |> 
  set_engine("nnet") |> 
  set_mode("classification")

model_set <-  workflow_set(
  preproc = list(normalized_recipe = titanic_recipe),
  models = list(glm = glm_spec,
                glmnet = glmnet_spec,
                rpart = rpart_spec, 
                ranger = rf_spec, 
                xgb = xgb_spec,
                nnet = nn_spec),
                cross = TRUE
  )

model_set

# Set up parallel processing
max_cores <- parallel::detectCores()
plan(multisession, workers = max_cores)

# Fit models (consider finetune::tune_race_anova for faster training)
ctrl <- control_grid(save_pred = TRUE, 
  save_workflow = TRUE,
  parallel_over = "everything")

tic(msg = "Many models trained with workflowsets in")
model_set_trained <- model_set |> 
  workflow_map(
    resamples = titanic_folds, 
    grid = 50, 
    seed = 12345,
    control = control_grid(save_pred = TRUE, 
                           save_workflow = TRUE,
                           parallel_over = "everything")
    )
toc()

# Select best of each method
autoplot_obj <- autoplot(model_set_trained, 
  select_best = TRUE,
  rank_metric = "roc_auc",
  metric = "roc_auc",
  type = "wflow_id")

wflow_ids <- model_set_trained |> 
  mutate(x = row_number()) |> 
  select(x, wflow_id)

autoplot_data <- ggplot_build(autoplot_obj)$data[[2]]

autoplot_data |>
  left_join(wflow_ids, by = join_by(group == x)) |> 
  ggplot(aes(x = x, y = y, colour = wflow_id)) +
  geom_point(size = 3) +
  geom_text(aes(y = y, label = paste0(round(y * 100, 1), "%")),
            vjust = -1.3) +
  geom_text(aes(y = y, label = wflow_id), 
            angle = 90, 
            hjust = 1.1) +
  scale_x_continuous(breaks = 1:7) +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0.70, 0.90)) +
  labs(y = "ROC AUC",
       x = "Rank Order") +
  theme(legend.position = "none")

# Rank results
ranks <- model_set_trained |> 
  rank_results(rank_metric = "roc_auc", select_best = TRUE) |> 
  select(rank, mean, model, wflow_id, .config)

top_performing_model <- ranks$wflow_id[[1]]

# Since Ranger was the best-performing model,
# extract it and select best-forming hyperparameters
model_set_tuning_wf <- extract_workflow(model_set_trained, top_performing_model)
model_set_results <- extract_workflow_set_result(model_set_trained, top_performing_model)
best_hyperparameters <- select_best(model_set_results, metric = "roc_auc")

final_wfs_workflow <- model_set_tuning_wf |> 
  finalize_workflow(parameters = best_hyperparameters)

# Last fit: This re-trains w/ entire training set and evaluates with initial testing split
set.seed(12345)
final_fit <- last_fit(final_wfs_workflow, titanic_split)

# Collect metrics from final_fit using test set for performance eval 
collect_metrics(final_fit)
  