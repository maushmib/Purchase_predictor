import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA, FactorAnalysis

def select_marketing_signals(X_train, y_train, X_test, feature_names=None, top_k=15):
    """
    Selects top features to optimize marketing signals.
    """
    k = min(top_k, X_train.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    best_features = []
    if feature_names is not None:
        mask = selector.get_support()
        best_features = np.array(feature_names)[mask].tolist()
        
    return X_train_sel, X_test_sel, best_features, selector

def build_personas(X_train, X_test):
    """
    Applies Factor Analysis to build latent human-readable personas,
    and returns PCA distributions.
    """
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    fa = FactorAnalysis(n_components=3)
    X_train_fa = fa.fit_transform(X_train)
    X_test_fa = fa.transform(X_test)

    return {
        "pca": (X_train_pca, X_test_pca, pca),
        "fa": (X_train_fa, X_test_fa, fa)
    }

def interpret_personas(fa_scores):
    """
    Persona Interpretation Engine (CRITICAL)
    Takes raw factor logic (3 factors) and maps it to actionable persona labels.
    """
    labels = []
    explanations = []
    
    for score in fa_scores:
        # Example heuristic thresholding:
        f1, f2, f3 = score[0], score[1], score[2]
        
        if f1 > 0.5:
            labels.append("High Intent Buyer")
            explanations.append("High concentration on product-related metrics and intent markers.")
        elif f2 > 0.5:
            labels.append("Price Sensitive / Cart Abandoner")
            explanations.append("High correlation with exit rates or seasonal special days (deal hunting).")
        else:
            labels.append("Casual Browser")
            explanations.append("Low duration and low overall depth. Informational intent only.")
            
    return labels, explanations
