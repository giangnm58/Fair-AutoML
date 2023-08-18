import sys
import os

# Get the directory path containing autosklearn
package_dir = os.path.abspath(os.path.join(os.path.dirname("Fair-AutoML"), '../..'))
# Add the directory to sys.path
sys.path.append(package_dir)
import datetime
import pickle

from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant
from sklearn.ensemble import RandomForestClassifier
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

train_list = "data_orig_train_german.pkl"
test_list = "data_orig_test_german.pkl"
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
temp_path = "german_svc_eod" + str(now)
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

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
import resource
import sys

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE
from autosklearn.pipeline.implementations.util import softmax
from autosklearn.util.common import check_for_bool, check_none
class CustomSVC(AutoSklearnClassificationAlgorithm):
    def __init__(self, C, kernel, gamma, shrinking, tol, max_iter,
                 class_weight=None, degree=3, coef0=0, random_state=42):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.svm

        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM. The cache size is
        # calculated as 2/3 of the available memory (which is calculated as the memory limit minus
        # the used memory)
        try:
            # Retrieve memory limits imposed on the process
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)

            if soft > 0:
                # Convert limit to units of megabytes
                soft /= 1024 * 1024

                # Retrieve memory used by this process
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024

                # In MacOS, the MaxRSS output of resource.getrusage in bytes; on other platforms,
                # it's in kilobytes
                if sys.platform == 'darwin':
                    maxrss = maxrss / 1024

                cache_size = (soft - maxrss) / 1.5

                if cache_size < 0:
                    cache_size = 200
            else:
                cache_size = 200
        except Exception:
            cache_size = 200

        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)

        self.shrinking = check_for_bool(self.shrinking)

        if check_none(self.class_weight):
            self.class_weight = None

        self.estimator = sklearn.svm.SVC(C=self.C,
                                         kernel=self.kernel,
                                         degree=self.degree,
                                         gamma=self.gamma,
                                         coef0=self.coef0,
                                         shrinking=self.shrinking,
                                         tol=self.tol,
                                         class_weight=self.class_weight,
                                         max_iter=self.max_iter,
                                         random_state=self.random_state,
                                         cache_size=cache_size,
                                         decision_function_shape='ovr')
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'LibSVM-SVC',
            'name': 'LibSVM Support Vector Classification',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': False,
            'handles_multioutput': False,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (PREDICTIONS,)}

    # SVC(C=0.85, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    #     max_iter=-1, random_state=42, shrinking=True, tol=0.001, probability=True,
    #     verbose=False)
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 0.04717, 11335, log=True,
                                       default_value=0.85)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(name="kernel",
                                           choices=["rbf", "poly", "sigmoid"],
                                           default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", 3, 4, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 3.1466158232736585e-05, 0.95334,
                                           log=True, default_value=0.1)
        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter("coef0", -0.59878, 0.51839, default_value=0.0)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                              default_value="True")
        tol = UniformFloatHyperparameter("tol", 1.2507352751797604e-05, 0.02704, default_value=1e-3,
                                         log=True)
        # cache size is not a hyperparameter, but an argument to the program!
        max_iter = UnParametrizedHyperparameter("max_iter", -1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                                tol, max_iter])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs


autosklearn.pipeline.components.classification.add_classifier(CustomSVC)
cs = CustomSVC.get_hyperparameter_search_space()
print(cs)
############################################################################
# Custom metrics definition
# =========================
def accuracy(solution, prediction):
    metric_id = 3
    protected_attr = 'sex'
    with open('test_split.txt') as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = data_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        default = sklearn.svm.LinearSVC(tol=0.001, C = 0.85, random_state =42)
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
    include_estimators=['CustomSVC'],
    ensemble_size=1,
    include_preprocessors=['extra_trees_preproc_for_classification', 'polynomial', 'feature_agglomeration'],
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

a_file = open("german_svc_eod" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_german_svc_eod" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()
count = 0
print(predictions)
print(y_test, len(predictions))
print("EOD-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(data_orig_test, predictions, 'sex'))
print(statistical_parity_difference(data_orig_test, predictions, 'sex'))
print(equal_opportunity_difference(data_orig_test, predictions, y_test, 'sex'))
print(average_odds_difference(data_orig_test, predictions, y_test, 'sex'))

