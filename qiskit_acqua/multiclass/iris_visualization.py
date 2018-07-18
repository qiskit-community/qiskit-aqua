from sklearn import datasets
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from sklearn.multiclass import OutputCodeClassifier
import numpy as np

# def temporary():
#     from sklearn import datasets
#     from sklearn.cross_validation import train_test_split
#     from sklearn.svm import LinearSVC
#     from qiskit_acqua.multiclass.dimension_reduction import reduce_dim_to
#     from qiskit_acqua.multiclass.AllPairs import AllPairs
#     from qiskit_acqua.multiclass_quantumsvm.QKernelSVM_Estimator import QKernalSVM_Estimator
#     from qiskit_acqua.multiclass.iris_dataset import deterministic_sample
#     iris = datasets.load_iris()
#     X, y = iris.data, iris.target
#     X_3d = reduce_dim_to(X, 2)
#     totalsize = len(X_3d)
#
#     X_train, y_train, X_test, y_test = deterministic_sample(X_3d, y)
#
#     cond = np.logical_or(y_train == 0, y_train == 1)
#     indcond = np.arange(y_train.shape[0])[cond]
#     X_train = X_train[indcond]
#     y_train = y_train[indcond]
#     y_train[y_train==0] = -1
#     y_train=y_train.astype(float) # to make sure cvxopt does not complain about the type!
#
#
#
#     cond = np.logical_or(y_test == 0, y_test == 1)
#     indcond = np.arange(y_test.shape[0])[cond]
#     X_test = X_test[indcond]
#     y_test = y_test[indcond]
#     y_test[y_test==0] = -1
#     y_test=y_test.astype(float) # to make sure cvxopt does not complain about the type!
#     return X_train, y_train, X_test, y_test




def deterministic_sample(X_3d, y,  train_size=20, test_size=20, train_seed=0, test_seed=59):
    totalsize = len(X_3d)

    np.random.seed(train_seed)
    indices = np.random.choice(totalsize, train_size, False)
    X_train = X_3d[indices]
    y_train = y[indices]
    # print(X_train, y_train)
    np.random.seed(test_seed)
    indices = np.random.choice(totalsize, test_size, False)
    X_test = X_3d[indices]
    y_test = y[indices]

    # print(X_test, y_test)
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # iris: each point has 4 features (sepal lengh/width and petal width/length) and  falls into one of the 3 classes.
    iris = datasets.load_iris()
    X, y = iris.data, iris.target


    def reduce_dim_to(X, dim):
        X_reduced = PCA(n_components=dim).fit_transform(X)
        return X_reduced

    ############# pca to 2d, then plot 2d###########
    X_2d = reduce_dim_to(X, 2)
    X_train,X_test,y_train,y_test=train_test_split(X_2d,y,test_size=0.5)
    def plot_2d(X, y):
        plt.figure(2, figsize=(8, 6))
        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y_train, cmap=plt.cm.Set1,
                    edgecolor='k')
        plt.xlabel('1st eigenvector')
        plt.ylabel('2nd eigenvector')
        plt.xticks(())
        plt.yticks(())
        plt.show()
    plot_2d(X_train, y_train)



    ############# pca to 3d, then plot 3d###########
    X_3d = reduce_dim_to(X, 3)
    X_train,X_test,y_train,y_test=train_test_split(X_3d,y,test_size=0.5)
    def plot_3d(X, y):
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
                   cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])
        plt.show()
    plot_3d(X_train, y_train)



    # def count_diff(A, B):
    #     l = len(A)
    #     diff = 0
    #     for i in range(0, l):
    #         if A[i] != B[i]:
    #             diff = diff + 1
    #     return diff


    # print(y_test)
    # one_against_all_result = (OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test))
    # print(count_diff(one_against_all_result, y_test))


    # all_pairs_result = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
    # print(count_diff(all_pairs_result, y_test))


    # clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
    # error_correcting_result = clf.fit(X_train, y_train).predict(X_test)
    # print(count_diff(error_correcting_result, y_test))


