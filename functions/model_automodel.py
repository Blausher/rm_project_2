import pandas as pd
import numpy as np

import copy

import datetime
from datetime import datetime

from tqdm.auto import tqdm, trange
from copy import copy, deepcopy

import sys
import warnings
warnings.simplefilter("ignore")
from tqdm.auto import tqdm


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
sns.set_style('whitegrid')

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import optuna
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.ERROR)


class Model:
    def __init__(self, n_components=None, pca_args=None, lr_args=None):
        self.pca_args = pca_args or {"random_state": 42}
        self.lr_args = lr_args or {}
        self.n_components = n_components
        self.pca = n_components and PCA(n_components, **self.pca_args)
        self.lr = LinearRegression(**self.lr_args)
        self.cols = None

    def get_values(self, x):
        if self.cols and isinstance(x, pd.DataFrame):
            x = x[self.cols]
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.values
        return x

    def fit(self, data, target):
        if isinstance(data, pd.DataFrame):
            self.cols = list(data.columns)
        data = self.get_values(data)
        target = self.get_values(target)
        if self.pca:
            data = self.pca.fit_transform(data)
        self.lr.fit(data, target)
        return self

    def predict_pca(self, data):
        data = self.get_values(data)
        if self.pca:
            return self.pca.transform(data)
        return data

    def predict_lr(self, data):
        return self.lr.predict(data)

    def predict(self, data):
        data = self.predict_pca(data)
        return self.predict_lr(data)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.pca, self.lr)


class AutoModel:
    def __init__(self, cv_splits=5, n_trials=10, max_components=10, metric=None):
        self.splitter = TimeSeriesSplit(cv_splits)
        self.study = optuna.study.create_study(sampler=optuna.samplers.TPESampler(seed=42), direction="maximize")
        self.metric = metric or r2_score
        self.n_trials = n_trials
        self.max_components = max_components

    def fit(self, data, target):
        data_ = data
        target_ = target
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(target, (pd.DataFrame, pd.Series)):
            target = target.values

        def objective(trial):
            components = trial.suggest_int("n_components", 2, min([self.max_components, *data.shape]))
            for t in trial.study.trials:
                if t.state != optuna.trial.TrialState.COMPLETE:
                    continue
                if t.params == trial.params:
                    raise optuna.exceptions.TrialPruned("Duplicate parameter set")

            scores = []
            for train_idx, test_idx in self.splitter.split(data):
                model = Model(components)
                model.fit(data[train_idx], target[train_idx])
                proba = model.predict(data[test_idx])
                score = self.metric(target[test_idx], proba)
                scores.append(score)
            return np.mean(scores)

        self.study.optimize(objective, self.n_trials)
        return Model(**self.study.best_params).fit(data_, target_)