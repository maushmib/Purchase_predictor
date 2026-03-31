from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(X_train, y_train, X_test):
    selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel