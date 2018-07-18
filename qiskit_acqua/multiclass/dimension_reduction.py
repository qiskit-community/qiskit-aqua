from sklearn.decomposition import PCA

def reduce_dim_to(X, dim):
    X_reduced = PCA(n_components=dim).fit_transform(X)
    return X_reduced
