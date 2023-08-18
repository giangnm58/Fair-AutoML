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

import autosklearn.pipeline.components.classification
from autosklearn.Fairea.fairea import create_baseline
from autosklearn.pipeline.components.classification \
    import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE
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

train_list = "data_orig_train_bank.pkl"
test_list = "data_orig_test_bank.pkl"
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
temp_path = "bank_rf_aod" + str(now)
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



# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

class CustomRandomForest(AutoSklearnClassificationAlgorithm):
    def __init__(self, n_estimators, criterion, max_features,
                  min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, max_depth=9, random_state=1, n_jobs=1,
                 class_weight=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.estimator = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier

        self.n_estimators = int(self.n_estimators)

        if check_none(self.max_depth):
            self.max_depth = 9
        else:
            self.max_depth = int(self.max_depth)

        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

        if self.max_features not in ("sqrt", "log2", "auto"):
            max_features = int(X.shape[1] ** float(self.max_features))
        else:
            max_features = self.max_features

        self.bootstrap = check_for_bool(self.bootstrap)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.min_impurity_decrease = float(self.min_impurity_decrease)

        # initial fit of only increment trees
        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            warm_start=True)
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
        return {'shortname': 'RF',
                'name': 'Random Forest Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        n_estimators = UniformIntegerHyperparameter("n_estimators", 207, 780, default_value=207)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="entropy")

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0.14647, 0.73536, default_value=0.5)

        max_depth = UnParametrizedHyperparameter("max_depth", 9)
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 6, 16, default_value=6)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 5, 16, default_value=5)
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True")
        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                bootstrap, min_impurity_decrease])
        return cs


#Add custom random forest classifier component to auto-sklearn.
autosklearn.pipeline.components.classification.add_classifier(CustomRandomForest)
cs = CustomRandomForest.get_hyperparameter_search_space()
print(cs)


############################################################################
# Custom metrics definition
# =========================

def accuracy(solution, prediction):
    metric_id = 4
    protected_attr = 'age'
    with open('test_split.txt') as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = data_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        default = RandomForestClassifier(n_estimators=50, criterion='entropy', max_features=0.5, min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, bootstrap=True, max_depth=9)
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
    include_estimators=['CustomRandomForest'],
    ensemble_size=1,
    include_preprocessors=['select_rates_classification', 'random_trees_embedding', 'feature_agglomeration'],
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
print(predictions)


a_file = open("bank_rf_aod" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_bank_rf_aod" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()

print(y_test, len(predictions))
print("AOD-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(data_orig_test, predictions, 'age'))
print(statistical_parity_difference(data_orig_test, predictions, 'age'))
print(equal_opportunity_difference(data_orig_test, predictions, y_test, 'age'))
print(average_odds_difference(data_orig_test, predictions, y_test, 'age'))

