from sklearn.decomposition import PCA, FactorAnalysis

def reduce_dimensionality(X_train, X_test):
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    fa = FactorAnalysis(n_components=3)
    X_train_fa = fa.fit_transform(X_train)
    X_test_fa = fa.transform(X_test)

    return {
        "pca": (X_train_pca, X_test_pca),
        "fa": (X_train_fa, X_test_fa),
    }