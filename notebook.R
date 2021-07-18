#' ---
#' title: 'Notebook'
#' output: github_document
#' ---
#' 
#' Typical EDA and modeling setup. My `import` function does two pretty notable things:
#' 
#' 1. Cleans the `ap_scores` columns (notably using `tidyr::separate`), creating `n/sum/avg_ap_score` columns.
#' 2. For the training set, it coerces `accepted` to a factor, which is expected by `{tidymodels}` for a classification task.
#' 
#' Notably, I used [an old blog post that I wrote](https://tonyelhabr.rbind.io/post/nested-json-to-tidy-data-frame-r/) to help me remember how to do the cleaning of `ap_scores`.
#' 
# /*
# knitr::spin('notebook.R', knit = FALSE, report = TRUE)
# */
#' 
#+ setup, echo=F, include=F ----
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE,
  fig.width = 8,
  fig.height = 8
)
#+ import ----
library(tidyverse)
library(tidymodels)
extrafont::loadfonts('win', quiet = TRUE)
library(tonythemes) # personal theme package
tonythemes::theme_set_tony()
# parallel::detectCores() # 8
doParallel::registerDoParallel(cores = 4)

import <- function(x) {
  df <-
    file.path(sprintf('%s.csv', x)) %>%
    read_csv() %>% 
    # This will be helpful for re-joining data after splitting out the `ap_scores`.
    mutate(idx = row_number()) %>% 
    relocate(idx)
  
  if(x == 'train') {
    df <-
      df %>% 
      mutate(across(accepted, ~ifelse(.x == 0, 'no', 'yes') %>% factor()))
  }
  
  # Find the max number of semicolons.
  rgx_split <- '\\;'
  n_col_max <-
    df %>% 
    pull(ap_scores) %>% 
    str_split(rgx_split) %>% 
    map_dbl(~length(.)) %>% 
    max()
  
  # Separate out columns, using the max number of semicolons.
  # Also, pivot to a tidy format and make some numberical features.
  nms_sep <- sprintf('ap_score_%02d', 1:n_col_max)
  df_sep <-
    df %>% 
    select(idx, ap_scores) %>% 
    separate(ap_scores, into = nms_sep, sep = rgx_split, fill = 'right') %>% 
    pivot_longer(
      -idx
    ) %>% 
    drop_na(value) %>% 
    group_by(idx) %>% 
    summarize(n_ap_score = n(), sum_ap_score = sum(as.integer(value))) %>% 
    ungroup() %>% 
    mutate(avg_ap_score = sum_ap_score / n_ap_score)
  df_sep
  
  res <-
    df %>% 
    select(-ap_scores) %>% 
    left_join(df_sep, by = 'idx') %>% 
    select(-idx)
  res
}

df <- import('train')
df_hold <- import('test') # I like to think of this more as a "holdout" data set.

#' `{skimr}` is almost always the first thing I do on a new data set. The `complete.cases` line is helpful to identify the rows with NAs.
#+ eda-1, results='asis' ----
df %>% skimr::skim()

#+ eda-1b ----
df %>% filter(!complete.cases(.))

#' Checking for disproportional `accepted`, which we'll want to know so that we can account for it by up/down-sampling later. It seems to be relatively balanced, so I don't think we'll do anything about it.
#+ eda-2 ----
df %>% count(accepted) %>% mutate(frac = n / sum(n))

#' `ethnicity` is the only categorical feature. We see that White, Asian, and Black are the largest groups by far. For modeling purposes, it may be useful to just group the rest of the groups together.
#+ eda-3 ----
df %>% count(ethnicity, sort = TRUE) %>% mutate(frac = n / sum(n))

#' Here I use the [Jeffrey's interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Jeffreys_interval), which is like a Bayesian take on a confidence interval for a binomial type of variable (`accepted`), to give some insight into the amount of error
#+ eda-4 ----
p_ethnicity <-
  df %>%
  group_by(ethnicity) %>%
  summarize(
    n = n(),
    n_accepted = sum(accepted == 'yes'),
    frac_accepted = n_accepted / n,
    low = qbeta(.025, n_accepted + .5, n - n_accepted + .5),
    high = qbeta(.975, n_accepted + .5, n - n_accepted + .5)
  ) %>% 
  ungroup() %>% 
  mutate(across(ethnicity, ~fct_reorder(.x, frac_accepted))) %>% 
  ggplot() +
  aes(x = frac_accepted, y = ethnicity) +
  geom_point(size = 3) +
  geom_errorbarh(position = position_dodge(width = 1), aes(xmin = low, xmax = high), height = 0.5) +
  scale_x_continuous(labels = percent) +
  theme(
    panel.grid.major.y = element_blank()
  ) +
  labs(
    title = str_wrap('How does acceptance rate differ by ethnicity? And how confident can we be in our empirical estimates?', 60),
    y = NULL,
    x = '% of Accepted'
  )
p_ethnicity

#' Correlations are a quick way to get some insight into our numeric features. `sat` and `gpa` are the most correlated features, which is not too surprising. Nonetheless, they are not super highly correlated, so I don't think we really have an issue with multicollinearity (which wouldn't be hard to account for anyways).
#+ cors ----
df_num <- 
  df %>% 
  select(accepted, where(is.numeric)) %>% 
  # more interested in avg and n than the sum, for modeling
  select(-sum_ap_score)
df_num

cors <-
  df_num %>%
  mutate(across(accepted, as.integer)) %>% 
  corrr::correlate(quiet = TRUE) %>% 
  pivot_longer(-term) %>% 
  rename(col1 = term, col2 = name, cor = value) %>% 
  filter(col1 != col2) %>% 
  arrange(desc(abs(cor)))
cors      

#' The correlations just for `accepted`... These are relatively low across the board, meaning that we may have some work to do. Looking at how `gpa` and `sat` are the most highly correlated, I think we should find that these will be the most important features in a linear model.
#+ cors_filt ----
cors_filt <-
  cors %>% 
  filter(col1== 'accepted') %>% 
  filter(col2 != 'accepted')
cors_filt

#' One thing I like to do with classification problems is to look at the distributions of the features given the response variable (`accepted`). Using percentiles for the x-axis can help prevent some distortion and clarify differences between the "yes"/"no" distributions.
#+ p_num ----
p_num <-
  df_num %>% 
  pivot_longer(-accepted) %>% 
  group_by(name) %>% 
  mutate(across(value, percent_rank)) %>% 
  ungroup() %>% 
  ggplot() +
  aes(x = value, fill = accepted) +
  geom_density(alpha = 0.5) +
  scale_x_continuous(labels = percent) +
  facet_wrap(~name, scales = 'free_y') +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = 'top'
  ) +
  labs(
    title = str_wrap('How are the distributions of numeric features different for accepted students vs. not accepted?', 60),
    y = 'Density',
    x = 'Percentile'
  )
p_num

#' I was trying to see if we could learn something from the interaction of the `n_ap_score` and `sum_ap_score` features. (Is there a self-selection bias? People who take more tests are more likely to have higher scores? Or maybe it's the other way around?) To be honest, I didn't really end up finding anything that seemed significant from this.
#+ p_score ----
p_score <-
  df %>% 
  count(n_ap_score, sum_ap_score, accepted) %>% 
  # drop_na() %>% 
  # mutate(across(matches('ap_scores'), factor)) %>% 
  ggplot() +
  aes(x = n_ap_score, y = sum_ap_score, color = accepted) +
  geom_jitter(aes(size = n), alpha = 0.5) +
  geom_smooth(method = 'loess', formula = formula(y ~ x)) +
  labs(
    title = 'Is there a selection bias with AP scores?',
    subtitle = 'Visually, it\'s inconclusive',
    x = '# of AP Scores',
    y = 'Sum of AP Scores'
  )
p_score

#' Modeling setup time.
#+ model-setup ----
set.seed(42)
split <- df %>% initial_split(strata = accepted)
df_trn <- split %>% training()
df_tst <- split %>% testing()
folds <- df_trn %>% vfold_cv(10)

rec <-
  df_trn %>% 
  recipe(formula(accepted ~ .), data = .) %>% 
  # this will be a linear combo of n_ and avg_, so drop it
  step_rm(sum_ap_score) %>% 
  # step_indicate_na(matches('ap_score')) %>% 
  step_impute_knn(
    ethnicity, sat, n_ap_score, avg_ap_score,
    impute_with = c(
      'gpa', 'essay_strength', 'family_income'
    )
  ) %>% 
  # group non-Black and -White ethnicities into one "other" group.
  step_other(ethnicity, threshold = 0.1) %>% 
  step_dummy(ethnicity)

# quick check
jui_trn <- rec %>% prep() %>% juice()
jui_trn

# log loss is most important imo (for probability), but let's look at several metrics
metset <- metric_set(mn_log_loss, accuracy, roc_auc)
ctrl <- control_grid(
  extract = extract_model,
  save_pred = TRUE,
  save_workflow = TRUE
)

#' Let's start with penalized linear regression (using `{glmnet}`). Due to the relatively simple nature of the data, this might just be all that we need. We'll explore gradient boosting (a tried and true out-of-the-box solution) later.
#+ model-glmnet ----
wf_lin <-
  rec %>%
  workflow(
    logistic_reg(
      'classification',
      engine = 'glmnet',
      # mixture = tune(), # alpha in glmnet, 1 by default (lasso)
      # mixture = 0.5, # hard-code elastic net
      # mixture = 1, # lasso
      penalty = tune() # lambda in glmnet
    )
  )

grid_lin <- crossing(
  mixture = c(0, 0.25, 0.5, 0.75, 1),
  # mixture = 0.5,
  penalty = 10 ^ seq(-3, 0, .1)
)
grid_lin

#+ tune-lin-pre, echo=F, include=F, eval=T
path_lin <- here::here('tune_lin.rds')
path_exists_lin <- file.exists(path_lin)

#+ tune-lin, eval=!path_exists_lin
# The underlying glmnet engine seems to change lambda sets without the random seed
set.seed(42)
tune_lin <-
  wf_lin %>% 
  tune_grid(
    folds,
    metrics = metset,
    control = ctrl,
    grid = grid_lin
  )

#+ tune-lin-post-export, echo=F, include=F, eval=!path_exists_lin
write_rds(tune_lin, path_lin)

#+ tune-lin-post-import, echo=F, include=F, eval=T
tune_lin <- read_rds(path_lin)

#+ autoplot-lin ----
tune_lin %>% 
  autoplot() + 
  theme(legend.position = 'top') +
  labs(
    title = 'Hyperparameter tuning for glmnet'
  )

#+ mets-lin ----
mets_lin <- tune_lin %>% collect_metrics()
mets_lin

#+ finalize-lin ----
params_best_lin <- tune_lin %>% select_best('mn_log_loss')
params_best_lin
wf_best_lin <- wf_lin %>% finalize_workflow(params_best_lin)
# Ugh, not working for some reason...
# fit_trn_lin <- wf_lin %>% fit(df_trn)
# fit_lin <- wf_lin %>% fit(df)

# Do this instead as a fix
wf_lin_fix <-
  rec %>%
  workflow(
    logistic_reg(
      'classification',
      engine = 'glmnet',
      penalty = params_best_lin$penalty, 
      mixture = params_best_lin$mixture
    )
  )

fit_trn_lin <- wf_lin_fix %>% fit(df_trn) # use this to evaluate validation set.
fit_lin <- wf_lin_fix %>% fit(df) # use this for true holdout data

#' Feature importances, a.k.a. coefficient estimates (since this is a linear model), are always great to look at. Notably, we see that `gpa` is the most important, but we don't see that `sat` is as important. Recall that these were the 2 most correlated features with `accepted`. However, they were also somewhat correlated with one another (~0.5), so perhaps it's not all that surprising to see that one of the two is treated with less importance by a regularized regression.
#+ imp-lin ----
imp_lin <-
  fit_lin %>% 
  extract_fit_engine() %>% 
  vip::vi(
    method = 'model', 
    lambda = params_best_lin$penalty, 
    alpha = params_best_lin$mixture
  ) %>% 
  set_names(c('feature', 'imp', 'sign')) %>% 
  mutate(across(feature, ~fct_reorder(.x, imp))) %>% 
  ggplot() +
  aes(x = imp, y = feature, fill = sign) +
  geom_col() +
  labs(
    title = 'glmnet feature importance',
    y = NULL,
    x = 'Coefficient'
  ) +
  theme(
    panel.grid.major.y = element_blank()
  )
imp_lin

#+ imp-lin-2, echo=F, include=F, eval=F
# Note that this is the same thing
fit_lin %>% 
  tidy(
    penalty = params_best_lin$penalty, 
    mixture = params_best_lin$mixture
  ) %>% 
  arrange(desc(abs(estimate)))

#' Get predictions for holdout set ("test.csv").
#+ preds-lin ----
# Save these for potential exporting, after checking how good xgboost is
probs_hold_lin <-
  fit_lin %>% 
  predict(df_hold, type = 'prob')
probs_hold_lin

#' Evaluate on validation set.
#+ probs-tst-lin ----
probs_tst_lin <- 
  fit_trn_lin %>% 
  augment(
    df_tst, 
    penalty = params_best_lin$penalty, 
    mixture = params_best_lin$mixture,
    type = 'prob'
  )

preds_tst_lin <- fit_trn_lin %>% augment(df_tst)

preds_tst_lin %>% 
  accuracy(.pred_class, accepted)

preds_tst_lin %>% 
  conf_mat(.pred_class, accepted) %>% 
  autoplot('heatmap') +
  labs(
    title = 'Confusion matrix for glmnet'
  )

# 'first' event level is the "no" class.
probs_tst_lin %>% 
  mn_log_loss(accepted, .pred_no, event_level = 'first')

probs_tst_lin %>% 
  roc_curve(accepted, .pred_no) %>% 
  autoplot() +
  tonythemes::theme_tony() +
  labs(
    title = 'ROC AUC for glmnet'
  )

#' Now let's follow the same process, but with `{xgboost}`.
#+ model-xg ----
wf_xg <-
  rec %>%
  # could definitely tune more here, but whatever
  workflow(
    boost_tree(
      'classification',
      engine = 'xgboost',
      trees = tune(),
      mtry = tune(),
      learn_rate = tune()
    )
  )

n_col_jui <- ncol(jui_trn)
# Regular grid, to make visual interpretation easier
# Actually had higher number of trees to begin with, but found that lower tended to better, so adjusted.
grid_xg <- crossing(
  trees = c(1, 5, 10, 25, 50, 75, 100),
  mtry = seq.int(2, n_col_jui, by = 2),
  learn_rate = c(0.01, 0.02, 0.03)
)

#+ tune-xg-prep, echo=F, include=F
path_xg <- here::here('tune_xg.rds')
path_exists_xg <- file.exists(path_xg)

#+ tune-xg, echo=F, include=F, eval=!path_exists_xg
tune_xg <-
  wf_xg %>% 
  tune_grid(
    folds,
    metrics = metset,
    control = ctrl,
    grid = grid_xg
  )
# beepr::beep(8) # mario

#+ tune-xg-post-export, echo=F, include=F, eval=!path_exists_xg
write_rds(tune_xg, path_xg)

#+ tune-xg-post-import, echo=F, include=F, eval=T
tune_xg <- read_rds(path_xg)

#+ autoplot-xg ----
tune_xg %>% 
  autoplot() + 
  theme(legend.position = 'top') +
  labs(
    title = 'Hyperparameter tuning for xgboost'
  )

#+ finalize-xg
params_best_xg <- tune_xg %>% select_best('mn_log_loss')
params_best_xg
wf_best_xg <- wf_xg %>% finalize_workflow(params_best_xg)
# Yay this pattern works for the xgboost model
fit_trn_xg <- wf_best_xg %>% fit(df_trn)
fit_xg <- wf_best_xg %>% fit(df)

#' Feature importances... we shouldn't expect these to be the same as those from glmnet.
#+ imp-xg ----
imp_xg <-
  fit_xg %>% 
  extract_fit_engine() %>% 
  # xgboost::xgb.importance() %>% # not working for some reason, ugh
  vip::vi(type = 'gain') %>% 
  set_names(c('feature', 'imp')) %>% 
  # arrange(desc(imp)) %>% 
  mutate(across(feature, ~fct_reorder(.x, imp))) %>% 
  ggplot() +
  aes(x = imp, y = feature) +
  geom_col() +
  labs(
    title = 'xgboost Feature Importance',
    y = NULL,
    x = 'Coefficient'
  ) +
  theme(
    panel.grid.major.y = element_blank()
  )
imp_xg

#' Evaluation time again. We find that the tuned xgboost is slightly worse than glmnet across all evaluation metrics.
#+ eval-xg ----
probs_tst_xg <- fit_trn_xg %>% augment(df_tst, type = 'prob')
preds_tst_xg <- fit_trn_xg %>% augment(df_tst)
preds_tst_xg %>% 
  accuracy(.pred_class, accepted)

preds_tst_lin %>% 
  conf_mat(.pred_class, accepted) %>% 
  autoplot('heatmap') +
  labs(
    title = 'Confusion matrix for xgboost'
  )

# 'first' event level is the "no" class.
probs_tst_xg %>% 
  mn_log_loss(accepted, .pred_no, event_level = 'first')

# This is slightly worse than the linear model
probs_tst_xg %>% 
  roc_curve(accepted, .pred_no) %>% 
  autoplot() +
  tonythemes::theme_tony() +
  labs(
    title = 'ROC AUC for xgboost'
  )

#' We could do some ensembling here. Perhaps the xgboost model captures some things that the penalized regression does not. However, I'm just about at the end of my time. Let's use the glmnet predictions.
#+ export-pred
write_csv(probs_hold_lin, file.path('probs.csv'))



