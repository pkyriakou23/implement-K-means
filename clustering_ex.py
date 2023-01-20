from pickle import TRUE
from keras.datasets import mnist
from matplotlib import pyplot
import math
from sklearn.naive_bayes import GaussianNB
import numpy as np

def Maximin(array, size, center_num, V):
    distance = [0.0] * size
    hcis = [0] * size
    centers = np.zeros((center_num, V))
    centers[0] = array[0]
    
    i = 0
    while i < size:
        sum = 0
        j = 0
        while j < V:
            sum = sum + (array[i][j] - centers[0][j]) ** 2
            j = j + 1
        distance[i] = math.sqrt(sum)
        hcis[i] = 0
        i = i + 1
    
    i = 1
    while i < center_num:
        index_max = 0
        dmax = 0.0
        j = 0
        while j < size:
            if distance[j] <= dmax:
                j = j + 1
                continue
            while hcis[j] < i - 1:
                hcis[j] = hcis[j] + 1
                another_sum = 0
                z = 0
                while z < V:
                    another_sum = another_sum + ((array[j][z] - centers[hcis[j]][z]) ** 2)
                    z = z + 1
                d = math.sqrt(another_sum)
                if d < distance[j]:
                    distance[j] = d
                    if d < dmax:
                        break

            if distance[j] > dmax:
                dmax = distance[j]
                index_max = j
            j = j + 1
        distance[index_max] = 0.0
        centers[i] = array[index_max]
        i = i + 1
    return centers


# this function calculate the new centers for the Kmeans algorithm 
def NewCenters(V, data):
    centers = np.zeros((1, V))
    i = 0
    while i < V:
        sum = 0
        j = 0
        while j < len(data):
            sum = sum + data[j][i]
            j = j + 1
        centers[0][i] = sum
        i = i + 1

    new = np.zeros((1, V))
    i = 0
    while i < V:
        new[0][i] = centers[0][i] / len(data)
        i = i + 1
    return new[0]


def Kmeans(centers, array, y_train, V, plot):

    data1 = []
    train1 = []
    data2 = []
    train2 = []
    data3 = []
    train3 = []
    data4 = []
    train4 = []
    d = [0.0] * len(centers)

    while True:
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        
        i = 0
        while i < len(array):
            j = 0
            while j < len(centers):
                sum = 0
                z = 0
                # euclidean distance
                while z < V:
                    sum = sum + ((array[i][z] - centers[j][z]) ** 2)
                    z = z + 1
                d[j] = math.sqrt(sum)
                j = j + 1
            #minimun distance
            min_d = min(d[0], d[1], d[2], d[3])
            if min_d == d[0]:
                data1.append([V])
                data1[x1] = array[i]
                x1 = x1+ 1
                train1.append(y_train[i])
            elif min_d == d[1]:
                data2.append([V])
                data2[x2] = array[i]
                x2 = x2 + 1
                train2.append(y_train[i])
            elif min_d == d[2]:
                data3.append([V])
                data3[x3] = array[i]
                x3 = x3 + 1
                train3.append(y_train[i])
            elif min_d == d[3]:
                data4.append([V])
                data4[x4] = array[i]
                x4 = x4 + 1
                train4.append(y_train[i])
            i = i + 1

        new = np.zeros((4, V))
        # find new centers 
        new[0] = NewCenters(V, data1)
        new[1] = NewCenters(V, data2)
        new[2] = NewCenters(V, data3)
        new[3] = NewCenters(V, data4)

        # if centers did not change is the answer else run the algorithm again
        if np.array_equal(centers, new):
            break
        else:
            centers = new
            data1.clear()
            train1.clear()
            data2.clear()
            train2.clear()
            data3.clear()
            train3.clear()
            data4.clear()
            train4.clear()
            d = [0.0] * len(centers)
    # find which of the digit appears most 
    # count the times tha appears
    sum1 = train1.count(max(train1, key=train1.count))
    sum2 = train2.count(max(train2, key=train2.count))
    sum3 = train3.count(max(train3, key=train3.count))
    sum4 = train4.count(max(train4, key=train4.count))

    purity = (sum1 + sum2 + sum3 + sum4) / (len(array))

    # [:,0] Rreturn the column 0 of the array(2D)
    # plot a variable to fill the data if we need to plot a graph
    if plot == True:
        data1x = np.array(data1)[:,0]
        data1y = np.array(data1)[:,1]
        data2x = np.array(data2)[:,0]
        data2y = np.array(data2)[:,1]
        data3x = np.array(data3)[:,0]
        data3y = np.array(data3)[:,1]
        data4x = np.array(data4)[:,0]
        data4y = np.array(data4)[:,1]
        pyplot.scatter(data1x, data1y, color='red')
        pyplot.scatter(data2x, data2y, color='green')
        pyplot.scatter(data3x, data3y, color='blue')
        pyplot.scatter(data4x, data4y, color='yellow')
        pyplot.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
        pyplot.show()
    
    return purity

#  Implement the Principal Component Analysis (PCA) algorithm.
def PCA(M, V):
    M2 = M - np.mean(M, axis=0)
    covariance = np.cov(M2, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    sorted = np.argsort(values)[::-1]
    sortedVectors = vectors[:, sorted]
    vectorSubset = sortedVectors[:, 0:V]
    reduced = np.dot(vectorSubset.transpose(), M2.transpose()).transpose()

    return reduced

def main():
  
    #load data from mnist where x image (as array), where y the answer
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    temp = []
#    temp_y= []
    i = 0
    # load only a subset of it, consisting of the classes (digits) i = 1, 3, 7, 9
    while i < len(train_y):
        if (train_y[i] == 1) | (train_y[i] == 3) | (train_y[i] == 7) | (train_y[i] == 9):
           #get the positions of 1,3,7,9
           temp.append(i); 
            #          temp_x.append(train_X[i])
            #          temp_y.append(train_y[i])
        i = i + 1
    
    #x y _train has only the 1 3 5 7 9
    x_train = train_X[temp]
    y_train = train_y[temp]
    temp = []
    i = 0
    #same for test
    while i < len(test_y):
        if (test_y[i] == 1) | (test_y[i] == 3) | (test_y[i] == 7) | (test_y[i] == 9):
            temp.append(i)
                    #            temp_x_test.append(test_X[i])
                    #           temp_y_test.append(test_y[i])
        i = i + 1
    
    X_test = test_X[temp]
    Y_test = test_y[temp]

            #    x_train = temp_x
            #    y_train = temp_y
            #    X_test = temp_x_test
            #    Y_test = temp_y_test

    
    #After calculating the two-dimensional features for each sample in M, use them to create a matrix M' 
    M = np.zeros((len(x_train), 2))
    
    i = 0 
    # the first feature component is the mean pixel value of all image matrix 
    # rows with odd index, while the second feature component is calculated as the mean pixel
    # value of all image matrix columns whose index is an even number  
    while i < (len(x_train)):
             
        odd = 0
        even = 0
        j = 0
        while j < 28:
            #check if it is odd or even
            if j % 2 != 0:
                z = 0
                while z < 28:
                    odd = odd + x_train[i][j][z]
                    z = z +1
            j = j + 1

        j = 0
        while j < 28:
            if j % 2 == 0:
                z = 0
                while z < 28:
                    even = even + x_train[i][z][j]
                    z = z + 1
            j = j +  1
        odd = odd / 392.0
        even = even / 392.0
        M[i][0] = odd
        M[i][1] = even
        i = i + 1

    
    #Use a scatter plot to visualize all rows of M' ... assign different colors for different class samples
    x1 = []
    y1 = []
    x3 = []
    y3 = []
    x7 = []
    y7 = []
    x9 = []
    y9 = []
    
    i = 0 
    # seperation of each class
    while i < (len(x_train)):
        if y_train[i] == 1:
            x1.append(M[i][0])
            y1.append(M[i][1])

        if y_train[i] == 3:
            x3.append(M[i][0])
            y3.append(M[i][1])

        if y_train[i] == 7:
            x7.append(M[i][0])
            y7.append(M[i][1])

        if y_train[i] == 9:
            x9.append(M[i][0])
            y9.append(M[i][1])
        i = i + 1 
             
    pyplot.scatter(x1, y1, color='red')
    pyplot.scatter(x3, y3, color='green')
    pyplot.scatter(x7, y7, color='blue')
    pyplot.scatter(x9, y9, color='yellow')
    pyplot.show()
    
    # use your implementation of the Maximin algorithm to initialize the cluster centers in the K-Means algorithm
    # 4 for centers and 2 for dimension
    centers = Maximin(M, len(x_train), 4, 2)
    purity = Kmeans(centers, M, y_train, 2, True)
    print('purity : ', purity)

    # 28*28 to 784
    M2 = np.zeros((len(x_train), 784), dtype=int)
    i = 0
    while i < len(x_train):
        j = 0
        while j < 784:
            z = 0
            while z < 28:
                k = 0
                while k < 28:
                    M2[i][j] = x_train[i][z][k]
                    j = j + 1
                    k = k + 1
                z = z + 1
        i = i + 1 

    # Aplly the algorithm to reduce the dimension of rows of M, in order to get a new matrix M˜ ,
    # where V = 2 25 50 100 the new number dimensions

    V = PCA(M2, 2)
    c2 = Maximin(V, len(V), 4, 2)
    purity = Kmeans(c2, V, y_train, 2, True)
    print('purity for V=2 is : ', purity)

    V25 = PCA(M2, 25)
    c3 = Maximin(V25, len(V25), 4, 25)
    purity = Kmeans(c3, V25, y_train, 25, False)
    print('purity for V=25 is : ', purity)

    V = PCA(M2, 50)
    c4 = Maximin(V, len(V), 4, 50)
    purity = Kmeans(c4, V, y_train, 50, False)
    print('purity for V=50 is : ', purity)

    V = PCA(M2, 100)
    c5 = Maximin(V, len(V), 4, 100)
    purity = Kmeans(c5, V, y_train, 100, False)
    print('purity for V=100 is : ', purity)
    
    #  Implement a Gaussian Naive Bayes Classifier. Use the rows of M˜ for V = Vmax and the
    # ground truth labels of Ltr to train the classifier. Then, use the same dimensionality reduction
    # process on the test data samples (rows of N), in order to obtain N˜ . Use the trained classifier
    # on the new test samples (rows of N˜ ). Use the classification results and Lte to calculate the
    # classification accuracy.
    
    classifier = GaussianNB()
    
    #Use the rows of M˜ for V = Vmax = 25
    #fit for training
    classifier.fit(V25, y_train)
    N = np.zeros((len(X_test), 784), dtype=int)
    i = 0
    while i < len(X_test):
        j = 0
        while j < 784:
            z = 0
            while z < 28:
                k = 0
                while k < 28:
                    N[i][j] = X_test[i][z][k]
                    j = j + 1
                    k = k + 1
                z = z + 1
            j = j + 1    
        i = i + 1
    V_25_test = PCA(N, 25)
    
    # score = the mean accuracy on the given test data and labels
    score = classifier.score(V_25_test, Y_test)
    print("Naive Bayes score: ", score)

if __name__ == "__main__":
    main()