# -*- encoding: utf-8 -*-
"""
=======
Metrics
=======

*Auto-sklearn* supports various built-in metrics, which can be found in the
:ref:`metrics section in the API <api:Built-in Metrics>`. However, it is also
possible to define your own metric and use it to fit and evaluate your model.
The following examples show how to use built-in and self-defined metrics for a
classification problem.
"""
import datetime
import pickle

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import autosklearn.pipeline.components.classification
from autosklearn.Fairea.fairea import create_baseline
from autosklearn.pipeline.components.classification \
    import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE, SIGNED_DATA
import shutil
import autosklearn.classification
import autosklearn.metrics
import warnings
warnings.filterwarnings('ignore')
from aif360.datasets import AdultDataset, BankDataset
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

import sklearn.metrics
import autosklearn.classification
from autosklearn.upgrade.metric import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference
from autosklearn.util.common import check_for_bool, check_none


train_list = "data_orig_train_bank5.pkl"
test_list = "data_orig_test_bank5.pkl"
def custom_preprocessing(df):
    def group_race(x):
        if x == "White":
            return 1.0
        else:
            return 0.0

    # Recode sex and race
    df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
    df['race'] = df['race'].apply(lambda x: group_race(x))

    return df

############################################################################
# File Remover
# ============
now = str(datetime.datetime.now())[:19]
now = now.replace(":","_")
temp_path = "bank_xgb_spd" + str(now)
try:
    os.remove("test_split.txt")
except:
    pass
try:
    os.remove("num_keys.txt")
except:
    pass
try:
    os.remove("beta.txt")
except:
    pass

f = open("beta.txt", "w")
f.close()
############################################################################
# Data Loading
# ============

import pandas as pd
from aif360.datasets import  StandardDataset

train = pd.read_pickle(train_list)
test = pd.read_pickle(test_list)

data_orig_train = StandardDataset(df=train, label_name='y',
            favorable_classes=['yes'],
            protected_attribute_names=['age'],
            privileged_classes=[lambda x: x >= 25],
            instance_weights_name=None,
            categorical_features=['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome'],
            features_to_keep=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx',
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome'], features_to_drop=[], na_values=["unknown"],
            custom_preprocessing=None, metadata=None)

data_orig_test = StandardDataset(df=test, label_name='y',
            favorable_classes=['yes'],
            protected_attribute_names=['age'],
            privileged_classes=[lambda x: x >= 25],
            instance_weights_name=None,
            categorical_features=['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome'],
            features_to_keep=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx',
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome'], features_to_drop=[], na_values=["unknown"],
            custom_preprocessing=None, metadata=None)

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

class CustomXGBoost(AutoSklearnClassificationAlgorithm):
    def __init__(self,
                 n_estimators,
                 max_depth,
                 learning_rate,
                 subsample,
                 min_child_weight,
                 random_state=None,
                 ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.estimator = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'XG',
                'name': 'XGBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': False,
                # Both input and output must be tuple(iterable)
                'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                'output': [PREDICTIONS]}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        n_estimators = UniformIntegerHyperparameter("n_estimators", 200, 761, default_value=200)
        max_depth = UniformIntegerHyperparameter("max_depth", 3, 9,
                                                 default_value=7)
        learning_rate = UniformFloatHyperparameter("learning_rate", 0.18411, 0.61907,
                                                   default_value=0.18411)
        subsample = UniformFloatHyperparameter("subsample", 0.32629, 0.75605,
                                               default_value=0.75)

        min_child_weight = UniformIntegerHyperparameter("min_child_weight", 6, 17,
                                                        default_value=6)

        cs.add_hyperparameters([n_estimators, max_depth, learning_rate, subsample,
                                min_child_weight])
        return cs


autosklearn.pipeline.components.classification.add_classifier(CustomXGBoost)
cs = CustomXGBoost.get_hyperparameter_search_space()
print(cs)

############################################################################
# Custom metrics definition
# =========================

def accuracy(solution, prediction):
    metric_id = 2
    protected_attr = 'age'
    with open('test_split.txt') as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = data_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        default = XGBClassifier()
        degrees = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        mutation_strategies = {"0": [1, 0], "1": [0, 1]}
        dataset_orig = subset_data_orig_train
        res = create_baseline(default, dataset_orig, privileged_groups, unprivileged_groups,
                              data_splits=10, repetitions=10, odds=mutation_strategies, options=[0, 1],
                              degrees=degrees)
        acc0 = np.array([np.mean([row[0] for row in res["0"][degree]]) for degree in degrees])
        acc1 = np.array([np.mean([row[0] for row in res["1"][degree]]) for degree in degrees])
        fair0 = np.array([np.mean([row[metric_id] for row in res["0"][degree]]) for degree in degrees])
        fair1 = np.array([np.mean([row[metric_id] for row in res["1"][degree]]) for degree in degrees])

        if min(acc0) > min(acc1):
            beta = (max(acc0) - min(acc0)) / (max(acc0) - min(acc0) + max(fair0))
        else:
            beta = (max(acc1) - min(acc1)) / (max(acc1) - min(acc1) + max(fair1))

        f = open("beta.txt", "w")
        f.write(str(beta))
        f.close()
    else:
        f = open("beta.txt", "r")
        beta = float(f.read())
        f.close()
        # print('yyyy')
    # print(beta)
    beta += 0.2
    if beta > 1.0:
        beta = 1.0
    try:
        num_keys = sum(1 for line in open('num_keys.txt'))
        print(num_keys)
        beta -= 0.050 * int(int(num_keys) / 10)
        if int(num_keys) % 10 == 0:
            os.remove(temp_path + "/.auto-sklearn/ensemble_read_losses.pkl")
        f.close()
    except FileNotFoundError:
        pass
    fairness_metrics = [1 - np.mean(solution == prediction),
                        disparate_impact(subset_data_orig_train, prediction, protected_attr),
                        statistical_parity_difference(subset_data_orig_train, prediction, protected_attr),
                        equal_opportunity_difference(subset_data_orig_train, prediction, solution, protected_attr),
                        average_odds_difference(subset_data_orig_train, prediction, solution, protected_attr)]

    print(fairness_metrics[metric_id], 1 - np.mean(solution == prediction),
          fairness_metrics[metric_id] * beta + (1 - np.mean(solution == prediction)) * (1 - beta), beta)

    return fairness_metrics[metric_id] * beta + (1 - np.mean(solution == prediction)) * (1 - beta)


############################################################################
# Second example: Use own accuracy metric
# =======================================
print("#"*80)
print("Use self defined accuracy metric")
accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu",
    score_func=accuracy,
    optimum=1,
    greater_is_better=False,
    needs_proba=False,
    needs_threshold=False,
)


############################################################################
# Build and fit a classifier
# ==========================
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60*60,
    # per_run_time_limit=500,
    memory_limit=10000000,
    include_estimators=['CustomXGBoost'],
    ensemble_size=1,
    include_preprocessors=['select_percentile_classification', 'select_rates_classification', 'fast_ica'],
    tmp_folder=temp_path,
    delete_tmp_folder_after_terminate=False,
    metric=accuracy_scorer
)
automl.fit(X_train, y_train)

###########################################################################
# Get the Score of the final ensemble
# ===================================

print(automl.show_models())
cs = automl.get_configuration_space(X_train, y_train)
print(cs)
predictions = automl.predict(X_test)

a_file = open("bank_xgb_spd" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_bank_xgb_spd" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()

print(predictions)
print(y_test, len(predictions))
print("SPD-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(data_orig_test, predictions, 'age'))
print(statistical_parity_difference(data_orig_test, predictions, 'age'))
print(equal_opportunity_difference(data_orig_test, predictions, y_test, 'age'))
print(average_odds_difference(data_orig_test, predictions, y_test, 'age'))


