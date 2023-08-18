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
import autosklearn.classification
import numpy as np
from aif360.datasets import GermanDataset

import sklearn.metrics
import autosklearn.classification
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



class CustomRandomForest(AutoSklearnClassificationAlgorithm):
    def __init__(self, n_estimators, criterion, max_features,
                  min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, max_depth=10, random_state=20, n_jobs=1,
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
            self.max_depth = None
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
        n_estimators = UniformIntegerHyperparameter("n_estimators", 280, 782, default_value=280)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0.13766, 0.73977, default_value=0.5)

        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 6, 16, default_value=6)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 6, 16, default_value=6)
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

        default = RandomForestClassifier(n_estimators=50, criterion='gini', max_features=0.5, min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, bootstrap=True, max_depth=None)
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
    include_preprocessors=['nystroem_sampler', 'kitchen_sinks', 'pca'],
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
count = 0
print(predictions)
print(y_test, len(predictions))
print("EOD-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(data_orig_test, predictions, 'sex'))
print(statistical_parity_difference(data_orig_test, predictions, 'sex'))
print(equal_opportunity_difference(data_orig_test, predictions, y_test, 'sex'))
print(average_odds_difference(data_orig_test, predictions, y_test, 'sex'))

a_file = open("german_rf_eod" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_german_rf_eod" + str(now) + "60sp.pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()

