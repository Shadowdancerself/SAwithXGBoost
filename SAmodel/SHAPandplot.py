"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate

folder_path = '/path/data'
for wid_size in range(4, 5):
    path = os.path.join(folder_path, f'{wid_size}.csv')
    data = pd.read_csv(path)
    data = data.fillna(data.mode().iloc[0])
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    random_seeds = [150]
    classifiers = {
        'SVM': SVC(),
        'RF': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'SR': LogisticRegression(multi_class='multinomial'),
        'XGB': xgb.XGBClassifier(objective='multi:softmax', num_class=10),
    }
    # hyper
    params = {
        'SVM': {'C': [0.1, 1, 10, 100],
                'gamma': [0.1, 1.5, 1, 10, 100, 1000],
                'kernel': ['rbf']
                },
        'RF': {'n_estimators': [100, 300, 500, 800, 1000],
               'max_features': [10, 11, 12],
               'min_samples_leaf': [2, 3, 4, 5]
               },
        'KNN': {'n_neighbors': [1, 3, 5, 7, 9, 11],
                'weights': ['distance'],
                'p': [1]
                },
        'SR': {'C': [0.1, 1, 10, 100],
               'max_iter': [100, 500, 1000],
               'solver': ['lsqr']
               },
        'XGB': {'n_estimators': [50, 100, 500, 1000],
                'learning_rate': [0.1, 0.01, 0.001],
                'max_depth': [2, 4, 6, 8, 10],
                'gamma': [0.5, 0.8, 1, 10, 100],
                'subsample': [0.1, 0.4, 0.5, 0.6],
                'reg_lambda': [0.001, 0.01],
                'min_child_weight': [5, 10, 15],
                'colsample_bytree': [0.8, 0.9],
                }
    }
    best_params = {}
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }
    for name, clf in classifiers.items():
        for seed in random_seeds:
            smote = SMOTE(random_state=seed)
            X, y = smote.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
            clf.random_state = seed
            print(f"{name}:Search Best Params")
            grid_search = GridSearchCV(clf, params[name], cv=inner_cv, refit="f1_weighted", n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_params[name] = grid_search.best_params_
            clf.set_params(**best_params[name])
            cv_results = cross_validate(clf, X_test, y_test, cv=outer_cv, scoring=scoring)
            print(f'Random seed {seed}, {name} Accuracy:', np.mean(cv_results['test_accuracy']))
            print(f'Random seed {seed}, {name} F1 score (weighted):', np.mean(cv_results['test_f1_weighted']))

            clf.fit(X_train, y_train)
            # 计算 SHAP 值
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_train)

            # 可视化 SHAP 值
            plt.figure(dpi=400)
            #shap.summary_plot(shap_values[1], X_train, plot_type="dot")  # "box" "bar"  "dot"
            shap.plots.waterfall(shap_values[0])
            plt.savefig("waterfall_plot.png")
