import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import statsmodels.stats.diagnostic
import statsmodels.api as sm


def load_data(filename, columnName):

    df = pd.read_csv(filename)
    df = df.fillna(0)
    ts = df[columnName]
    data = ts.values.reshape(-1, 1).astype("float32")  # (N, 1)
    print("time series shape:", data.shape)
    return ts, data

def createSamples(dataset, lookBack, RNN=True):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        sample_X = dataset[i:(i + lookBack), :]
        sample_Y = dataset[i + lookBack, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, 1)
    dataY = np.array(dataY)  # (N, 1)
    if not RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]))

    return dataX, dataY


# 分割时间序列作为样本,支持多步预测
def create_multi_ahead_samples(ts, lookBack, lookAhead=1, RNN=True):

    '''
    :param ts: input ts np array
    :param lookBack: history window size
    :param lookAhead: forecasting window size
    :param RNN: if use RNN input format
    :return: trainx with shape (sample_num, look_back, 1) or and trainy with shape (sample_num, look_ahead)
    '''

    dataX, dataY = [], []
    for i in range(len(ts) - lookBack - lookAhead):
        history_seq = ts[i: i + lookBack]
        future_seq = ts[i + lookBack: i + lookBack + lookAhead]
        dataX.append(history_seq)
        dataY.append(future_seq)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    if not RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]))
        dataY = np.reshape(dataY, (dataY.shape[0], dataY.shape[1]))
    return dataX, dataY


def divideTrainTest(dataset, rate=0.75):

    train_size = int(len(dataset) * rate)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:]
    return train, test



def createVariableDataset(dataset, minLen, maxLen, step):

    dataNum = len(dataset)
    X, Y = [], []

    for i in range(maxLen, len(dataset)):
        for lookBack in range(minLen, maxLen + 1, step):  # 遍历所有长度
            sample_X = dataset[i-lookBack:i]
            sample_Y = dataset[i]
            X.append(sample_X)
            Y.append(sample_Y)
    X = np.array(X)
    Y = np.array(Y)
    X = pad_sequences(X, maxlen=maxLen, dtype='float32')  # 左端补齐
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y



def transform_groundTruth(vtestY, minLen, maxLen, step):

    lagNum = (maxLen - minLen) // step + 1
    print("lag num is", lagNum)
    truth = []
    for i in range(0, len(vtestY), lagNum):
        truth.append(np.mean(vtestY[i:i + lagNum]))
    return np.array(truth)



def createPaddedDataset(dataset, lookBack, maxLen):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        a = dataset[i:(i + lookBack)]
        dataX.append(a)
        dataY.append(dataset[i + lookBack])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataX = pad_sequences(dataX, maxlen=maxLen, dtype='float32')  # 左端补齐
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
    return dataX, dataY



def align(trTrain,trTest,trendWin,resTrain,resTest,resWin):
    empWin = np.empty((trendWin, 1))
    empWin[:] = np.nan
    empWin2 = np.empty((resWin, 1))
    empWin2[:] = np.nan
    trendPred = np.vstack((empWin, trTrain))
    trendPred = np.vstack((trendPred, empWin))
    trendPred = np.vstack((trendPred, trTest))

    resPred = np.vstack((empWin2, resTrain))
    resPred = np.vstack((resPred, empWin2))
    resPred = np.vstack((resPred, resTest))

    return trendPred, resPred


def plot(trainPred, trainY, testPred, testY):
    pred = np.concatenate((trainPred, testPred))
    gtruth = np.concatenate((trainY, testY))
    plt.plot(pred, 'g')
    plt.plot(gtruth, 'r')
    plt.show()


def LBtest(data):
    r, q, p = sm.tsa.acf(data, qstat=True)
    data1 = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data1, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))


if __name__ == "__main__":

    ts, data = load_data("EUR-USD.csv", columnName="rate")
    train, test = divideTrainTest(data, 0.75)
    print("train num:", train.shape)
    print("test num:", test.shape)

    trainX, trainY = create_multi_ahead_samples(train, lookBack=24, lookAhead=6, RNN=True)
    testX, testY = create_multi_ahead_samples(test, lookBack=24, lookAhead=6, RNN=True)
    print("trainX shape is", trainX.shape)
    print("trainY shape is", trainY.shape)
    print("testX shape is", testX.shape)
    print("testY shape is", testY.shape)




