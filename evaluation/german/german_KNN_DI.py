import datetime
import pickle

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import autosklearn.pipeline.components.classification
from autosklearn.Fairea.fairea import create_baseline
from autosklearn.pipeline.components.classification \
    import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE, SIGNED_DATA
import autosklearn.classification
import numpy as np
from aif360.datasets import GermanDataset

import sklearn.metrics
import autosklearn.classification
from autosklearn.pipeline.implementations.util import softmax
from autosklearn.upgrade.metric import disparate_impact, statistical_parity_difference, equal_opportunity_difference,average_odds_difference
import os
import shutil

from autosklearn.util.common import check_for_bool, check_none

train_list = "data_orig_train_german10.pkl"
test_list = "data_orig_test_german10.pkl"
def custom_preprocessing(df):
    def group_credit_hist(x):
        if x in ['A30', 'A31', 'A32']:
            return 'None/Paid'
        elif x == 'A33':
            return 'Delay'
        elif x == 'A34':
            return 'Other'
        else:
            return 'NA'

    def group_employ(x):
        if x == 'A71':
            return 'Unemployed'
        elif x in ['A72', 'A73']:
            return '1-4 years'
        elif x in ['A74', 'A75']:
            return '4+ years'
        else:
            return 'NA'

    def group_savings(x):
        if x in ['A61', 'A62']:
            return '<500'
        elif x in ['A63', 'A64']:
            return '500+'
        elif x == 'A65':
            return 'Unknown/None'
        else:
            return 'NA'

    def group_status(x):
        if x in ['A11', 'A12']:
            return '<200'
        elif x in ['A13']:
            return '200+'
        elif x == 'A14':
            return 'None'
        else:
            return 'NA'

    status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                  'A92': 0.0, 'A95': 0.0}
    df['sex'] = df['personal_status'].replace(status_map)

    # group credit history, savings, and employment
    df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
    df['savings'] = df['savings'].apply(lambda x: group_savings(x))
    df['employment'] = df['employment'].apply(lambda x: group_employ(x))
    df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
    df['status'] = df['status'].apply(lambda x: group_status(x))
    df['credit'] = df['credit'].replace({2: 0.0, 1: 1.0})

    return df

############################################################################
# File Remover
# ============
now = str(datetime.datetime.now())[:19]
now = now.replace(":","_")
temp_path = "german_knn_aod" + str(now)
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
from aif360.datasets import GermanDataset, StandardDataset

train = pd.read_pickle(train_list)
test = pd.read_pickle(test_list)
na_values=[]
default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Old', 0.0: 'Young'}],
}
data_orig_train = StandardDataset(df=train, label_name='credit',
            favorable_classes=[1],
            protected_attribute_names=['sex'],
            privileged_classes=[[1]],
            instance_weights_name=None,
            categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker'],
            features_to_keep=['age', 'sex', 'employment', 'housing', 'savings', 'credit_amount', 'month', 'purpose'],
            features_to_drop=['personal_status'], na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=default_mappings)

data_orig_test = StandardDataset(df=test, label_name='credit',
            favorable_classes=[1],
            protected_attribute_names=['sex'],
            privileged_classes=[[1]],
            instance_weights_name=None,
            categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker'],
            features_to_keep=['age', 'sex', 'employment', 'housing', 'savings', 'credit_amount', 'month', 'purpose'],
            features_to_drop=['personal_status'], na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=default_mappings)

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]


X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()


# dataset_orig = GermanDataset(protected_attribute_names=['sex'],
#                             privileged_classes=[[1]],
#                             features_to_keep=['age', 'sex', 'employment', 'housing', 'savings', 'credit_amount', 'month', 'purpose'],
#                             custom_preprocessing=custom_preprocessing)
# privileged_groups = [{'sex': 1}]
# unprivileged_groups = [{'sex': 0}]
#
# data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)
#
# X_train = data_orig_train.features
# y_train = data_orig_train.labels.ravel()
#
# X_test = data_orig_test.features
# y_test = data_orig_test.labels.ravel()

# from sklearn.preprocessing import StandardScaler
#
# Scaler_X = StandardScaler()
# X_train = Scaler_X.fit_transform(X_train)
# X_test = Scaler_X.transform(X_test)

class CustomKNN(AutoSklearnClassificationAlgorithm):
    def __init__(self, n_neighbors, weights, p, random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state

    def fit(self, X, Y):
        import sklearn.neighbors
        import sklearn.multiclass

        estimator = \
            sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                                   weights=self.weights,
                                                   p=self.p)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
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
        return {'shortname': 'KNN',
                'name': 'K-Nearest Neighbor Classification',
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

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=11, upper=65, log=True, default_value=11)
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        cs.add_hyperparameters([n_neighbors, weights, p])

        return cs


autosklearn.pipeline.components.classification.add_classifier(CustomKNN)
cs = CustomKNN.get_hyperparameter_search_space()
print(cs)
############################################################################
# Custom metrics definition
# =========================
def accuracy(solution, prediction):
    metric_id = 1
    protected_attr = 'sex'
    with open('test_split.txt') as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = data_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        from sklearn.neighbors import KNeighborsClassifier

        default = KNeighborsClassifier(n_neighbors=10)
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
    include_estimators=['CustomKNN'],
    ensemble_size=1,
    include_preprocessors=['extra_trees_preproc_for_classification', 'select_percentile_classification', 'select_percentile_classification'],
    tmp_folder=temp_path,
    delete_tmp_folder_after_terminate=False,
    metric=accuracy_scorer
)
automl.fit(X_train, y_train)

###########################################################################
# Get the Score of the final ensemble
# ===================================

print(automl.show_models())
predictions = automl.predict(X_test)

a_file = open("german_knn_di" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_german_knn_di" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()
count = 0
print(predictions)
print(y_test, len(predictions))
print("DI-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(data_orig_test, predictions, 'sex'))
print(statistical_parity_difference(data_orig_test, predictions, 'sex'))
print(equal_opportunity_difference(data_orig_test, predictions, y_test, 'sex'))
print(average_odds_difference(data_orig_test, predictions, y_test, 'sex'))

