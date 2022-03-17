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
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE, SIGNED_DATA
from autosklearn.util.common import check_for_bool, check_none
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import autosklearn.classification
from autosklearn.upgrade.metric import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference
import os, shutil



train_list = "data_orig_train1.pkl"
test_list = "data_orig_test1.pkl"
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
now = now.replace(":", "_")
temp_path = "titanic_xgb_di" + str(now)
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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test.loc[:, 'Survived'] = 0

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List, Union, Dict


class SelectCols(TransformerMixin):
    """Select columns from a DataFrame."""

    def __init__(self, cols: List[str]) -> None:
        self.cols = cols

    def fit(self, x: None) -> "SelectCols":
        """Nothing to do."""
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return just selected columns."""
        return x[self.cols]


sc = SelectCols(cols=['Sex', 'Survived'])
sc.transform(train.sample(5))


class LabelEncoder(TransformerMixin):
    """Convert non-numeric columns to numeric using label encoding.
    Handles unseen data on transform."""

    def fit(self, x: pd.DataFrame) -> "LabelEncoder":
        """Learn encoder for each column."""
        encoders = {}
        for c in x:
            v, k = zip(pd.factorize(x[c].unique()))
            encoders[c] = dict(zip(k[0], v[0]))

        self.encoders_ = encoders

        return self

    def transform(self, x) -> pd.DataFrame:
        """For columns in x that have learned encoders, apply encoding."""
        x = x.copy()
        for c in x:
            # Ignore new, unseen values
            x.loc[~x[c].isin(self.encoders_[c]), c] = np.nan
            # Map learned labels
            x.loc[:, c] = x[c].map(self.encoders_[c])

        # Return without nans
        return x.fillna(-2).astype(int)


le = LabelEncoder()
le.fit_transform(train[['Pclass', 'Sex']].sample(5))


class NumericEncoder(TransformerMixin):
    """Remove invalid values from numerical columns, replace with median."""

    def fit(self, x: pd.DataFrame) -> "NumericEncoder":
        """Learn median for every column in x."""
        self.encoders_ = {
            c: pd.to_numeric(x[c],
                             errors='coerce').median(skipna=True) for c in x}

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # Create a list of new DataFrames, each with 2 columns
        output_dfs = []
        for c in x:
            new_cols = pd.DataFrame()
            # Find invalid values that aren't nans (-inf, inf, string)
            invalid_idx = pd.to_numeric(x[c].replace([-np.inf, np.inf],
                                                     np.nan),
                                        errors='coerce').isnull()

            # Copy to new df for this column
            new_cols.loc[:, c] = x[c].copy()
            # Replace the invalid values with learned median
            new_cols.loc[invalid_idx, c] = self.encoders_[c]
            # Mark these replacement in a new column called
            # "[column_name]_invalid_flag"
            new_cols.loc[:, f"{c}_invalid_flag"] = invalid_idx.astype(np.int8)

            output_dfs.append(new_cols)

        # Concat list of output_dfs to single df
        df = pd.concat(output_dfs,
                       axis=1)

        return df.fillna(0)


ne = NumericEncoder()
ne.fit_transform(train[['Age', 'Fare']].sample(5))

# LabelEncoding fork: Select object columns -> label encode
pp_object_cols = Pipeline([('select', SelectCols(cols=['Sex', 'Survived',
                                                       'Cabin', 'Ticket',
                                                       'SibSp', 'Embarked',
                                                       'Parch', 'Pclass',
                                                       'Name'])),
                           ('process', LabelEncoder())])

# NumericEncoding fork: Select numeric columns -> numeric encode
pp_numeric_cols = Pipeline([('select', SelectCols(cols=['Age',
                                                        'Fare'])),
                            ('process', NumericEncoder())])

# We won't use the next part, but typically the pipeline would continue to
# the model (after dropping 'Survived' from the training data, of course).
# For example:
pp_pipeline = FeatureUnion([('object_cols', pp_object_cols),
                            ('numeric_cols', pp_numeric_cols)])

model_pipeline = Pipeline([('pp', pp_pipeline),
                           ('mod', LogisticRegression())])
train_ = train

# .fit_transform on train
train_pp = pd.concat((pp_numeric_cols.fit_transform(train_),
                      pp_object_cols.fit_transform(train_)),
                     axis=1)

# .transform on test
test_pp = pd.concat((pp_numeric_cols.transform(test),
                     pp_object_cols.transform(test)),
                    axis=1)
test_pp.sample(5)

target = 'Survived'
x_columns = [c for c in train_pp if c != target]
x_train, y_train = train_pp[x_columns], train_pp[target]
x_test = test_pp[x_columns]

df = pd.concat((x_train, y_train), axis=1)
train = pd.read_pickle(train_list)
test = pd.read_pickle(test_list)
data_orig_train = StandardDataset(train,
                               label_name='Survived',
                               protected_attribute_names=['Sex'],
                               favorable_classes=[1],
                               privileged_classes=[[1]])
data_orig_test = StandardDataset(test,
                               label_name='Survived',
                               protected_attribute_names=['Sex'],
                               favorable_classes=[1],
                               privileged_classes=[[1]])

privileged_groups = [{'Sex': 1}]
unprivileged_groups = [{'Sex': 0}]

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()

# dataset_orig = StandardDataset(df,
#                                        label_name='Survived',
#                                        protected_attribute_names=['Sex'],
#                                        favorable_classes=[1],
#                                        privileged_classes=[[1]])
#
# privileged_groups = [{'Sex': 1}]
# unprivileged_groups = [{'Sex': 0}]
#
# data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)
#
# X_train = data_orig_train.features
# y_train = data_orig_train.labels.ravel()
#
# X_test = data_orig_test.features
# y_test = data_orig_test.labels.ravel()

class CustomXGBoost(AutoSklearnClassificationAlgorithm):
    def __init__(self,
                 n_estimators,
                 max_depth,
                 learning_rate,
                 subsample,
                 min_child_weight,
                 seed=0,
                 random_state=None
                 ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.seed = seed
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.estimator = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            min_child_weight=self.min_child_weight,
            seed=self.seed,
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
        n_estimators = UniformIntegerHyperparameter("n_estimators", 186, 684, default_value=300)
        max_depth = UniformIntegerHyperparameter("max_depth", 3, 9,
                                                 default_value=4)
        learning_rate = UniformFloatHyperparameter("learning_rate", 0.08833, 0.64799,
                                                   default_value=0.08833)
        subsample = UniformFloatHyperparameter("subsample", 0.31001, 0.81135,
                                               default_value=0.81135)

        min_child_weight = UniformIntegerHyperparameter("min_child_weight", 5, 16,
                                                        default_value=5)

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
    metric_id = 1
    protected_attr = 'Sex'
    with open('test_split.txt') as f:
        first_line = f.read().splitlines()
        last_line = first_line[-1]
        split = list(last_line.split(","))
    for i in range(len(split)):
        split[i] = int(split[i])

    subset_data_orig_train = data_orig_train.subset(split)

    if os.stat("beta.txt").st_size == 0:

        import xgboost as xgb
        default = xgb.XGBClassifier(learning_rate=0.01, max_depth=4, n_estimators=300, seed=0)
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
print("#" * 80)
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
    include_preprocessors=['select_percentile_classification', 'select_rates_classification', 'liblinear_svc_preprocessor'],
    tmp_folder=temp_path,
    delete_tmp_folder_after_terminate=False,
    metric=accuracy_scorer
)
automl.fit(X_train, y_train)

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

print(automl.show_models())

###########################################################################
# Get the Score of the final ensemble
# ===================================
a_file = open("titanic_xgb_di_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl.cv_results_, a_file)
a_file.close()

a_file1 = open("automl_titanic_xgb_di_60sp" + str(now) + ".pkl", "wb")
pickle.dump(automl, a_file1)
a_file1.close()
predictions = automl.predict(X_test)
count = 0
for i in predictions:
    if i == 0:
        count += 1
print(count, len(predictions))
print("DI-Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
print(disparate_impact(data_orig_test, predictions, 'Sex'))
print(statistical_parity_difference(data_orig_test, predictions, 'Sex'))
print(equal_opportunity_difference(data_orig_test, predictions, y_test, 'Sex'))
print(average_odds_difference(data_orig_test, predictions, y_test, 'Sex'))

