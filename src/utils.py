import matplotlib.pyplot as plt
import seaborn as sns

def plot_pca(X_pca, y):
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y)
    plt.title("PCA Scatter Plot")
    plt.show()