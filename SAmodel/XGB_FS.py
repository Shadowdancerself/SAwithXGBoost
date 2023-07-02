"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn

python: feature subsets result
"""

import pandas as pd
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
from openpyxl import Workbook
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate

folder_path = '/path/data'
for wid_size in range(4, 5):
    path = os.path.join(folder_path, f'{wid_size}.csv')
    print(path)
    data = pd.read_csv(path)
    data = data.fillna(data.mode().iloc[0])
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    random_seeds = [10, 20, 30, 400, 50, 175, 134, 330, 44, 82, 95, 200, 100, 66, 150, 38, 42, 5, 70, 60]
    repeats = 1
    # model
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
    model = []
    accuracy_scores = []
    f1_scores = []
    e_accuracy_scores = []
    e_f1_scores = []
    t_accuracy_scores = []
    t_f1_scores = []
    c_accuracy_scores = []
    c_f1_scores = []
    et_accuracy_scores = []
    et_f1_scores = []
    tc_accuracy_scores = []
    tc_f1_scores = []
    ce_accuracy_scores = []
    ce_f1_scores = []
    test_accuracy = []
    test_f1_weighted = []
    best_params = {}
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }

    for name, clf in classifiers.items():
        for seed in random_seeds:
            smote = SMOTE(random_state=seed)
            X, y = smote.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            clf.random_state = seed
            for i in range(repeats):
                print(f"{name}:Search Best Params")
                grid_search = GridSearchCV(clf, params[name], cv=inner_cv, refit="f1_weighted", n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_params[name] = grid_search.best_params_
                clf.set_params(**best_params[name])
                cv_results = cross_validate(clf, X_test, y_test, cv=outer_cv, scoring=scoring)
                print(f'Random seed {seed}, {name} Accuracy:', np.mean(cv_results['test_accuracy']))
                print(f'Random seed {seed}, {name} F1 score (weighted):', np.mean(cv_results['test_f1_weighted']))
                test_accuracy.append(np.mean(cv_results['test_accuracy']))
                test_f1_weighted.append(np.mean(cv_results['test_f1_weighted']))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                print(f'{name} Accuracy: {accuracy_scores[-1]:.4f}')

                X_test = pd.DataFrame(X_test)
                all_columns = X_test.columns


                e = all_columns[:23]
                t = all_columns[-8:]
                c = all_columns[19:23]
                tc = all_columns[-4:]


                X_e_train = X_train.drop(e, axis=1)
                X_t_train = X_train.drop(t, axis=1)
                X_c_train = X_train.iloc[:, c]
                X_et_train = X_train.drop(c, axis=1)
                X_tc_train = X_train.drop(tc, axis=1)
                X_ce_train = X_train.iloc[:, t]


                X_e = X_test.drop(e, axis=1)
                X_t = X_test.drop(t, axis=1)
                X_c = X_test.iloc[:, c]
                X_et = X_test.drop(c, axis=1)
                X_tc = X_test.drop(tc, axis=1)
                X_ce = X_test.iloc[:, t]

                # environment
                clf.fit(X_e_train, y_train)
                y_e = clf.predict(X_e)
                e_accuracy_scores.append(accuracy_score(y_test, y_e))
                e_f1_scores.append(f1_score(y_test, y_e, average='weighted'))
                print(f'{name} Accuracy: {e_accuracy_scores[-1]:.4f}')

                # eye tracking
                clf.fit(X_t_train, y_train)
                y_t = clf.predict(X_t)
                t_accuracy_scores.append(accuracy_score(y_test, y_t))
                t_f1_scores.append(f1_score(y_test, y_t, average='weighted'))
                print(f'{name} Accuracy: {t_accuracy_scores[-1]:.4f}')

                # car
                clf.fit(X_c_train, y_train)
                y_c = clf.predict(X_c)
                c_accuracy_scores.append(accuracy_score(y_test, y_c))
                c_f1_scores.append(f1_score(y_test, y_c, average='weighted'))
                print(f'{name} Accuracy: {c_accuracy_scores[-1]:.4f}')

                # environment + eye tracking
                clf.fit(X_et_train, y_train)
                y_et = clf.predict(X_et)
                et_accuracy_scores.append(accuracy_score(y_test, y_et))
                et_f1_scores.append(f1_score(y_test, y_et, average='weighted'))
                print(f'{name} Accuracy: {et_accuracy_scores[-1]:.4f}')

                # eye tracking + car
                clf.fit(X_tc_train, y_train)
                y_tc = clf.predict(X_tc)
                tc_accuracy_scores.append(accuracy_score(y_test, y_tc))
                tc_f1_scores.append(f1_score(y_test, y_tc, average='weighted'))
                print(f'{name} Accuracy: {tc_accuracy_scores[-1]:.4f}')

                # car + environment
                clf.fit(X_ce_train, y_train)
                y_ce = clf.predict(X_ce)
                ce_accuracy_scores.append(accuracy_score(y_test, y_ce))
                ce_f1_scores.append(f1_score(y_test, y_ce, average='weighted'))
                print(f'{name} Accuracy: {ce_accuracy_scores[-1]:.4f}')

                model.append(name)
                df = pd.DataFrame({
                    'model': model,
                    'Accuracy': accuracy_scores,
                    'F1 Score': f1_scores,
                    'e_accuracy_scores': e_accuracy_scores,
                    'e_f1_scores': e_f1_scores,
                    't_accuracy_scores': t_accuracy_scores,
                    't_f1_scores': t_f1_scores,
                    'c_accuracy_scores': c_accuracy_scores,
                    'c_f1_scores': c_f1_scores,
                    'et_accuracy_scores': et_accuracy_scores,
                    'et_f1_scores': et_f1_scores,
                    'tc_accuracy_scores': tc_accuracy_scores,
                    'tc_f1_scores': tc_f1_scores,
                    'test_accuracy': test_accuracy,
                    'test_f1_weighted': test_f1_weighted,
                })

                book = Workbook()
                writer = pd.ExcelWriter(f'aaa/{wid_size}/14s.xlsx', engine='openpyxl')
                writer.book = book
                df.to_excel(writer, sheet_name='Results', index=False)
                writer.save()
                if name == '':
                    plt.figure(figsize=(10, 8))
                    feat_importances = pd.Series(clf.feature_importances_, index=data.columns[1:])
                    feat_importances.sort_values(ascending=False).plot(kind='bar')
                    plt.title(f'{name} Feature Importance')
                    # plt.ylim(0, 0.5)
                    plt.ylabel('Percentage')
                    plt.savefig(f'output/{wid_size}/{name}_feature_importance.png')
                    plt.clf()

                    print(1)
                    features = []
                    # acc_permuted = []
                    # f1_permuted = []
                    # acc_permuted_std = []
                    # f1_permuted_std = []
                    accuracy_scores = []
                    f1_scores = []
                    accuracy_scores_std = []
                    f1_scores_std = []

                    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
                    # importance = result.importances_mean
                    for i in range(X.shape[1]):
                        features.append(i)
                        important_features = X[:, result.importances_mean.argsort()[::-1][:i + 1]]
                        # X_permuted = X.copy()
                        # X_permuted[:, i] = np.random.permutation(X[:, i])
                        # y_pred_permuted = clf.predict(X_permuted)
                        scores = cross_validate(clf, important_features, y, cv=5, scoring=('accuracy', 'f1_macro'))
                        accuracy_scores.append(scores['test_accuracy'].mean())
                        f1_scores.append(scores['test_f1_macro'].mean())
                        accuracy_scores_std.append(scores['test_accuracy'].std())
                        f1_scores_std.append(scores['test_f1_macro'].std())
                        # acc_permuted.append(accuracy_score(y, y_pred_permuted))
                        # f1_permuted.append(f1_score(y, y_pred_permuted, average='weighted'))
                        # acc_permuted_std.append(np.std(acc_permuted))
                        # f1_permuted_std.append(np.std(f1_permuted))

                    df = pd.DataFrame({
                        'features': features,
                        'accuracy_scores': accuracy_scores,
                        'f1_scores': f1_scores,
                        'accuracy_scores_std': accuracy_scores_std,
                        'f1_scores_std': f1_scores_std
                    })
                    book = Workbook()
                    writer = pd.ExcelWriter(f'output/{wid_size}/A_and_fi important_features.xlsx', engine='openpyxl')
                    writer.book = book
                    df.to_excel(writer, sheet_name='features', index=False)
                    writer.save()
                else:
                    print(1)




