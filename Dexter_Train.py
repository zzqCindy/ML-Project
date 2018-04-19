
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
        if len(line) == 1: line += ['']
        xi = zeros((1, 20000)) * nan
        for e in line:
            ind, val = e.split(":")
            xi[0][int(ind) - 1] = float(val)
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

    x = readData('F:\\program\\ML_Project\\DataSet\\DEXTER\\dexter_train.data')
    y = readLabel('F:\\program\\ML_Project\\DataSet\\DEXTER\\dexter_train.labels')

    print(shape(x))
