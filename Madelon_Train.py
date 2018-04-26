from numpy import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import *
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

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


def preprocess(x_train,y_train,x_valid,num,per):

    x = vstack((x_train,x_valid))

    # Preprocessing
    # Removing features with low variance
    sel = VarianceThreshold(threshold=(num))
    x = sel.fit_transform(x)
    data = sel.transform(x_train)
    test = sel.transform(x_valid)

    # Standard
    #minMax = MinMaxScaler().fit(x)
    #data = minMax.transform(data)
    #test = minMax.transform(test)
    #standard = StandardScaler().fit(x)
    #data = standard.transform(data)
    #test = standard.transform(test)

    # Feature Selection
    # mutual information
    #mutualInfo = SelectPercentile(mutual_info_classif,percentile=.99)
    #mutualInfo.fit(data,y_train)
    #data = mutualInfo.transform(data)
    #test = mutualInfo.transform(test)
    #print(shape(data))

    # PCA
    pca = PCA(n_components=per)
    data = pca.fit_transform(data)
    test = pca.transform(test)
    print(shape(data))

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
    w=np.random.rand(len(input_data[0]))#随机生成[0,1)之间,作为初始化w
    bias=0.0#偏置

    for i in range(iteration):
        samples= zip(input_data,y)
        for (input_i,label) in samples:#对每一组样本
            #计算f(w*xi+b),此时x有两个
            result=input_i*w+bias
            result=float(sum(result))
            y_pred=float(unit_step(result))#计算输出值 y^
            w=w+rate*(label-y_pred)*np.array(input_i)#更新权重
            bias=rate*(label-y_pred)#更新bias

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
    x_train = readData('.\\DataSet\\MADELON\\madelon_train.data')
    y_train = readLabel('.\\DataSet\\MADELON\\madelon_train.labels')

    x_valid = readData('.\\DataSet\\MADELON\\madelon_valid.data')
    y_valid = readLabel('.\\DataSet\\MADELON\\madelon_valid.labels')


    x = vstack((x_train,x_valid))

    # plot data
    plt.figure(1)
    plt.bar(range(len(var(x,axis=0))),var(x,axis=0))
    plt.ylim(0,2000)
    plt.xlabel('#features')
    plt.ylabel('variance')
    #plt.show()

    plt.figure(2)
    plt.bar(range(len(mean(x,axis=0))),mean(x,axis=0))
    plt.xlabel('#features')
    plt.ylabel('mean')
    #plt.show()

    for num in range(100,201,50):
        print('num = ',num)
        for per in range(90,99):
            per = float(per)/100
            print('per = ', per)
            data,test = preprocess(x_train,y_train,x_valid,num,per)
            print('KNN')
            KNN(data,y_train,test,y_valid)
            #linear
            #print('Perceptron')
            #perceptron(data,y_train,test,y_valid,100,.2)

            #gussian
            #print('Bayes')
            #bayes(data,y_train,test,y_valid)


    # preprocessing
    #data,test = preprocess(x_train,y_train,x_valid)

    #classifier
    #print('KNN')
    #KNN(x_train,y_train,x_valid,y_valid)
    #KNN(data,y_train,test,y_valid)

    #linear
    #print('Perceptron')
    #perceptron(x_train,y_train,x_train,y_train,100,.2)
    #perceptron(x_train,y_train,x_valid,y_valid,100,.2)
    #perceptron(data,y_train,test,y_valid,100,.2)

    #gussian
    #print('Bayes')
    #bayes(x_train,y_train,x_train,y_train)
    #bayes(x_train,y_train,x_valid,y_valid)
    #bayes(data,y_train,test,y_valid)
