
from numpy import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn import svm


def readData(addr):
    prob_x = array([])
    for line in open(addr):
        line = line.split(None)
        # In case an instance with all zero features
        if len(line) == 0: line += ['']
        xi = zeros((1, 500)) * 0
        ind = 0
        for e in line:
            xi[0][ind] = float(e)
            ind = ind+1
        if prob_x.size == 0:
            prob_x = xi
        else:
            prob_x = vstack((prob_x, xi))
    return prob_x


def readLabel(addr):
    prob_y = []
    for line in open(addr):
        label = line.strip()
        prob_y += [float(label)]
    return prob_y


if __name__ == "__main__":

    # read data
    x_train = readData('.\\DataSet\\MADELON\\madelon_train.data')
    y_train = readLabel('.\\DataSet\\MADELON\\madelon_train.labels')

    x_valid = readData('.\\DataSet\\MADELON\\madelon_valid.data')
    y_valid = readLabel('.\\DataSet\\MADELON\\madelon_valid.labels')

    # Preprocessing

    min_max = MinMaxScaler().fit(vstack((x_train,x_valid)))
    data = min_max.transform(x_train)
    test = min_max.transform(x_valid)

    # Feature Selection
    # Pearson Correlation



    # PCA

    pca = PCA(n_components=.95)
    data = pca.fit_transform(data)
    test = pca.transform(test)
    print(shape(data))

    # Classification
    clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovr')
    clf.fit(data, y_train)
    print(clf.score(data, y_train))
    print(clf.score(test, y_valid))





