
from numpy import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import *
import matplotlib.pyplot as plt
from sklearn import neighbors

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


def fisher_criterion(v1, v2):
    return np.power((np.mean(v1) - np.mean(v2)),2) / (np.var(v1) + np.var(v2))



if __name__ == "__main__":

    # read data
    x_train = readData('.\\DataSet\\MADELON\\madelon_train.data')
    y_train = readLabel('.\\DataSet\\MADELON\\madelon_train.labels')

    x_valid = readData('.\\DataSet\\MADELON\\madelon_valid.data')
    y_valid = readLabel('.\\DataSet\\MADELON\\madelon_valid.labels')

    x = vstack((x_train,x_valid))

    # Preprocessing

    plt.bar(range(len(var(x,axis=0))),var(x,axis=0))
    plt.ylim(0,500)
    #plt.show()

    # Removing features with low variance
    sel = VarianceThreshold(threshold=(500))
    x = sel.fit_transform(x)
    data = sel.transform(x_train)
    test = sel.transform(x_valid)
    print(shape(data))

    # Standard
    standard = StandardScaler().fit(x)
    data = standard.transform(data)
    test = standard.transform(test)

    # Feature Selection
    # mutual information
    mutualInfo = SelectPercentile(mutual_info_classif,percentile=90)
    mutualInfo.fit(data,y_train)
    data = mutualInfo.transform(data)
    test = mutualInfo.transform(test)
    print(shape(data))

    # PCA
    pca = PCA(n_components=.90)
    data = pca.fit_transform(data)
    test = pca.transform(test)
    print(shape(data))

    # Classification
    # SVM
    #clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
    #clf.fit(data, y_train)
    #print(clf.score(data, y_train))
    #print(clf.score(test, y_valid))

    # KNN
    knn = neighbors.KNeighborsClassifier(30)
    knn.fit(data,y_train)
    print(knn.score(data,y_train))
    print(knn.score(test,y_valid))




