
from numpy import *
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import *
from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

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


def preprocess(x_train,x_valid,ans,num):

    indexArray = []
    for index in range(0,20000):
        if ans[index,0] >= num:
            indexArray += [int(index)]
    print(shape(indexArray))

    data = x_train[:,indexArray[0]][:,np.newaxis]
    for index in range(1,len(indexArray)):
        data = np.concatenate((data,x_train[:,indexArray[index]][:,np.newaxis]),axis=1)

    test = x_valid[:,indexArray[0]][:,np.newaxis]
    for index in range(1,len(indexArray)):
        test = np.concatenate((test,x_valid[:,indexArray[index]][:,np.newaxis]),axis=1)

    # replace nan with 0
    whereNan = isnan(test)
    test[whereNan] = 0
    whereNan = isnan(data)
    data[whereNan] = 0

    return data, test


def KNN(data,y_train,test,y_valid):

    # Classification
    # KNN
    knn = neighbors.KNeighborsClassifier(30)
    knn.fit(data,y_train)
    #print(knn.score(data,y_train))
    print(knn.score(test,y_valid))


def perceptron(input_data,y,input_test,y_label,iteration,rate):

    unit_step = lambda x: -1 if x < 0 else 1
    w=np.random.rand(len(input_data[0]))#random w
    bias=0.0#bias

    for i in range(iteration):
        samples= zip(input_data,y)
        for (input_i,label) in samples:
            result=input_i*w+bias
            result=float(sum(result))
            y_pred=float(unit_step(result))#compute output y
            w=w+rate*(label-y_pred)*np.array(input_i)#update weight
            bias=rate*(label-y_pred)#update bias

    y_pred = []
    for input in input_test:
        result = input*w +bias
        result = sum(result)
        y_pred += [float(unit_step(result))]

    cor = correctRate(y_pred,y_label)
    print(cor)


def bayes(x_train,y_train,x_valid,y_valid):
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    y_pred = []
    for input in x_valid:
        result = clf.predict(input.reshape(1,-1))
        y_pred += [float(result)]

    cor = correctRate(y_pred,y_valid)
    print(cor)


def correctRate(y_pred,y_label):
    count = 0
    for index in range(0,len(y_label)):
        if not y_pred[index] == y_label[index]:
            count += 1
    return 1-(count/len(y_label))


if __name__ == "__main__":

    # read data
    x_train = readData('.\\DataSet\\DEXTER\\dexter_train.data')
    y_train = readLabel('.\\DataSet\\DEXTER\\dexter_train.labels')

    x_valid = readData('.\\DataSet\\DEXTER\\dexter_valid.data')
    y_valid = readLabel('.\\DataSet\\DEXTER\\dexter_valid.labels')

    x = vstack((x_train,x_valid))

    ans = np.zeros((20000,1))
    for item in x:
        for i in range(0,20000):
            if not np.isnan(item[i]):
                ans[i,0] += 1

    plt.bar(range(len(ans[:,0])),ans[:,0])
    plt.xlabel('features')
    plt.ylabel('number of valid data')
    plt.ylim(0,50)
    #plt.show()

    for num in range(10,21,2):

        print('num = ', num)
        data,test = preprocess(x_train,x_valid,ans,num)

        #KNN
        print('KNN')
        KNN(data,y_train,test,y_valid)

        #linear
        print('Perceptron')
        perceptron(data,y_train,test,y_valid,100,.2)

        #gussian
        print('Bayes')
        bayes(data,y_train,test,y_valid)



