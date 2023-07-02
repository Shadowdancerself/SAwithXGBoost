"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, make_scorer, \
    roc_auc_score, cohen_kappa_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from openpyxl import Workbook
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from Macro_Micro_AUC_kappa import y_pred_prob

folder_path = '/path/data'
for wid_size in range(1, 40):
        path = os.path.join(folder_path, f'{wid_size}.csv')
        data = pd.read_csv(path)
        data = data.fillna(data.mode().iloc[0])
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        # 10-fold nested cross-validation
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
        precision_scores = []
        f1_scores = []
        confusion_matrices = []
        recall_scores = []
        weighted_precision = []
        weighted_recall = []
        test_accuracy = []
        test_f1_weighted = []
        auc = []
        kappa = []
        macro_precision = []
        macro_recall = []
        macro_f1 = []
        micro_precision = []
        micro_recall = []
        micro_f1 = []
        best_params = {}
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_weighted': make_scorer(f1_score, average='weighted')
        }
        for name, clf in classifiers.items():
            model = name
            print(model)
            for seed in random_seeds:
                # SMOTE
                smote = SMOTE(random_state=seed)
                X, y = smote.fit_resample(X, y)
                # train :test 9：1
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)

                clf.random_state = seed
                for i in range(repeats):
                    print(f"{name}:Search Best Params")
                    grid_search = GridSearchCV(clf, params[name], cv=inner_cv, refit="f1_weighted", n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    best_params[name] = grid_search.best_params_
                    print(f"{name}:Best_Params: {best_params}")
                    clf.set_params(**best_params[name])
                    cv_results = cross_validate(clf, X_test, y_test, cv=outer_cv, scoring=scoring)
                    print(f'Random seed {seed}, {name} Accuracy:', np.mean(cv_results['test_accuracy']))
                    print(f'Random seed {seed}, {name} F1 score (weighted):', np.mean(cv_results['test_f1_weighted']))
                    test_accuracy.append(np.mean(cv_results['test_accuracy']))
                    test_f1_weighted.append(np.mean(cv_results['test_f1_weighted']))
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accuracy_scores.append(accuracy_score(y_test, y_pred))
                    print(f'{name} Accuracy: {accuracy_scores[-1]:.4f}')
                    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                    print(f'{name} f1: {f1_scores[-1]:.4f}')
                    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
                    print(f'{name} recall: {recall_scores[-1]:.4f}')
                    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
                    print(f'{name} precision: {precision_scores[-1]:.4f}')
                    weighted_precision.append(precision_score(y_test, y_pred, average='weighted'))
                    print(f'{name} weighted precision: {weighted_precision[-1]:.4f}')
                    weighted_recall.append(recall_score(y_test, y_pred, average='weighted'))
                    print(f'{name} weighted recall: {weighted_recall[-1]:.4f}')
                    confusion_matrices.append(confusion_matrix(y_test, y_pred))
                    auc.append(roc_auc_score(y_test, y_pred_prob, multi_class='ovo'))
                    kappa.append(cohen_kappa_score(y_test, y_pred))
                    print("AUC: ", np.mean(auc))
                    print("Cohen's kappa: ", np.mean(kappa))

                    if not os.path.exists(f'output/{wid_size}'):
                        os.makedirs(f'output/{wid_size}')

                    labels = sorted(set(y_test))
                    cm = confusion_matrix(y_test, y_pred, labels=labels)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(cm, cmap='Blues')
                    cbar = ax.figure.colorbar(im, ax=ax)
                    ax.set_xticks(np.arange(len(labels)))
                    ax.set_yticks(np.arange(len(labels)))
                    ax.set_xticklabels(labels)
                    ax.set_yticklabels(labels)
                    ax.set_xlabel('Predicted label')
                    ax.set_ylabel('True label')
                    ax.set_title('Confusion Matrix')
                    thresh = cm.max() / 2
                    for i in range(len(labels)):
                        for j in range(len(labels)):
                            ax.text(j, i, format(cm[i, j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black")
                    fig.tight_layout()
                    plt.savefig(f'output/{wid_size}/{name}_confusion_matrix.png', dpi=400)
                    plt.clf()
                    model.append(name)
                    df = pd.DataFrame({
                        'model': model,
                        'Accuracy': accuracy_scores,
                        'F1 Score': f1_scores,
                        'Recall': recall_scores,
                        'Precision': precision_scores,
                        'Confusion Matrix': confusion_matrices,
                        'weighted_precision': weighted_precision,
                        'weighted_recall': weighted_recall,
                        'test_accuracy': test_accuracy,
                        'test_f1_weighted': test_f1_weighted,
                    })
                    book = Workbook()
                    writer = pd.ExcelWriter(f'output/{wid_size}/results.xlsx', engine='openpyxl')
                    writer.book = book
                    df.to_excel(writer, sheet_name='Results', index=False)
                    writer.save()

                    if name == 'XGB':
                        plt.figure(figsize=(10, 8))
                        feat_importances = pd.Series(clf.feature_importances_, index=data.columns[1:])
                        feat_importances.sort_values(ascending=False).plot(kind='bar')
                        plt.title(f'{name} Feature Importance')
                        plt.ylabel('Percentage')
                        plt.savefig(f'output/{wid_size}/{name}_feature_importance.png')
                        plt.clf()
                        print(1)
                        features = []
                        accuracy_scores = []
                        f1_scores = []
                        accuracy_scores_std = []
                        f1_scores_std = []

                        result = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
                        for i in range(X.shape[1]):
                            features.append(i)
                            important_features = X[:, result.importances_mean.argsort()[::-1][:i + 1]]
                            X_permuted = X.copy()
                            X_permuted[:, i] = np.random.permutation(X[:, i])
                            y_pred_permuted = clf.predict(X_permuted)
                            scores = cross_validate(clf, important_features, y, cv=5, scoring=('accuracy', 'f1_macro'))
                            accuracy_scores.append(scores['test_accuracy'].mean())
                            f1_scores.append(scores['test_f1_macro'].mean())
                            accuracy_scores_std.append(scores['test_accuracy'].std())
                            f1_scores_std.append(scores['test_f1_macro'].std())

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
                        print(2)




