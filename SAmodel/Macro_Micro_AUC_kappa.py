"""
SA model prediction
Author : zhenghongtao
email: zhenghongtao@sjtu.edu.cn
"""


for train_index, test_index in outer_cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=inner_cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    y_pred_prob = grid_search.predict_proba(X_test)
    macro_precision.append(precision_score(y_test, y_pred, average='macro'))
    macro_recall.append(recall_score(y_test, y_pred, average='macro'))
    macro_f1.append(f1_score(y_test, y_pred, average='macro'))
    micro_precision.append(precision_score(y_test, y_pred, average='micro'))
    micro_recall.append(recall_score(y_test, y_pred, average='micro'))
    micro_f1.append(f1_score(y_test, y_pred, average='micro'))
    auc.append(roc_auc_score(y_test, y_pred_prob, multi_class='ovo'))
    kappa.append(cohen_kappa_score(y_test, y_pred))

print("Macro precision: ", np.mean(macro_precision))
print("Macro recall: ", np.mean(macro_recall))
print("Macro F1 score: ", np.mean(macro_f1))
print("Micro precision: ", np.mean(micro_precision))
print("Micro recall: ", np.mean(micro_recall))
print("Micro F1 score: ", np.mean(micro_f1))
print("AUC: ", np.mean(auc))
print("Cohen's kappa: ", np.mean(kappa))
results = pd.DataFrame({
    'Metric': ['Macro precision', 'Macro recall', 'Macro F1 score', 'Micro precision', 'Micro recall', 'Micro F1 score', 'AUC', 'Cohen\'s kappa'],
    'Value': [macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, auc, kappa]
})

book = Workbook()
writer = pd.ExcelWriter(f'mmak.xlsx', engine='openpyxl')
writer.book = book
results.to_excel(writer, sheet_name='Results', index=False)
writer.save()



