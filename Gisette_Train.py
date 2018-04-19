
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

    # read data
    x_train = readData('.\\DataSet\\GISETTE\\gisette_train.data')
    y_train = readLabel('.\\DataSet\\GISETTE\\gisette_train.labels')

    x_valid = readData('.\\DataSet\\GISETTE\\gisette_valid.data')
    y_valid = readLabel('.\\DataSet\\GISETTE\\gisette_valid.labels')


    print(x)
    print(y)
