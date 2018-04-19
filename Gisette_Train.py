
from numpy import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn import svm


def readData(addr):
    prob_x = array([])
    count = 1
    for line in open(addr):
        line = line.split(None)
        print('line',count)
        count += 1
        # In case an instance with all zero features
        if len(line) == 0: line += ['']
        xi = zeros((1, 5000)) * 0
        ind = 0
        for e in line:
            if not float(e) == 0:
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

    x = readData('F:\\program\\ML_Project\\DataSet\\GISETTE\\gisette_train.data')
    y = readLabel('F:\\program\\ML_Project\\DataSet\\GISETTE\\gisette_train.labels')

    print(x)
    print(y)
