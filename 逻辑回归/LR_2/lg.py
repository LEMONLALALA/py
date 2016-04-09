#coding:utf-8

from numpy import *
import matplotlib.pyplot as plt
import time


# sigmoid函数
def sigmoid(inX):
        return 1.0 / (1 + exp(-inX))


# 使用可选优化算法(optional optimize algorithm)来训练数据
# 输入： train_x 是一个矩阵，每一行代表一个样本　
#             train_y 是一个矩阵，每一行是相应的类别标签
#             opts 是一个优化选项，包括步长和迭代次数的最大值 
def trainLogRegres(train_x, train_y, opts):
        # 计算训练时间
        startTime = time.time()

        numSamples, numFeatures = shape(train_x)
        alpha = opts['alpha']; maxIter = opts['maxIter']
        weights = ones((numFeatures, 1))

        #用梯度下降算法优化
        for k in range(maxIter):
                if opts['optimizeType'] == 'gradDescent': # 梯度下降　
                        output = sigmoid(train_x * weights)
                        error = train_y - output
                        weights = weights + alpha * train_x.transpose() * error
                elif opts['optimizeType'] == 'stocGradDescent': #随机梯度下降
                        for i in range(numSamples):
                                output = sigmoid(train_x[i, :] * weights)
                                error = train_y[i, 0] - output
                                weights = weights + alpha * train_x[i, :].transpose() * error
                elif opts['optimizeType'] == 'smoothStocGradDescent': # 光滑的随机梯度下降
                        #随机选择样本，减少系数的周期性波动　
                        dataIndex = range(numSamples)
                        for i in range(numSamples):
                                alpha = 4.0 / (1.0 + k + i) + 0.01
                                randIndex = int(random.uniform(0, len(dataIndex)))
                                output = sigmoid(train_x[randIndex, :] * weights)
                                error = train_y[randIndex, 0] - output
                                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                                del(dataIndex[randIndex]) # during one interation, delete the optimized sample
                else:
                        raise NameError('Not support optimize method type!')


        print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
        return weights


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
        numSamples, numFeatures = shape(test_x)
        matchCount = 0
        for i in xrange(numSamples):
                predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
                if predict == bool(test_y[i, 0]):
                        matchCount += 1
        accuracy = float(matchCount) / numSamples
        return accuracy


# 用２维数据表现训练的逻辑回归模型　
def showLogRegres(weights, train_x, train_y):
    　
        numSamples, numFeatures = shape(train_x)
        if numFeatures != 3:
                print "Sorry! I can not draw because the dimension of your data is not 2!"
                return 1

        # 在图上画出所有的样本
        for i in xrange(numSamples):
                if int(train_y[i, 0]) == 0:
                        plt.plot(train_x[i, 1], train_x[i, 2], 'or')
                elif int(train_y[i, 0]) == 1:
                        plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

        # 画出分界线
        min_x = min(train_x[:, 1])[0, 0]
        max_x = max(train_x[:, 1])[0, 0]
        weights = weights.getA()  # 把矩阵转化为数组
        y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
        y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
        plt.xlabel('X1'); plt.ylabel('X2')
        plt.show()

if __name__ == '__main__':
     
    from numpy import *
    import matplotlib.pyplot as plt
    import time
    def loadData():
        train_x = []
        train_y = []
        fileIn = open('/home/darou/文档/桌面/LALALA/逻辑回归/LR_3/testSet.txt')
        for line in fileIn.readlines():
                lineArr = line.strip().split()
                train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
                train_y.append(float(lineArr[2]))
        return mat(train_x), mat(train_y).transpose()


# step 1: 读入数据
    print "step 1: 读入数据......"
    train_x, train_y = loadData()
    test_x = train_x; test_y = train_y

# step 2: 训练
    print "step 2: 训练......."
    opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
    optimalWeights = trainLogRegres(train_x, train_y, opts)

# step 3: 测试
    print "step 3: 测试......"
    accuracy = testLogRegres(optimalWeights, test_x, test_y)

# step 4:结果
    print "step 4: 结果"    
    print '正确率: %.3f%%' % (accuracy * 100)
    print '2维图像：'
    showLogRegres(optimalWeights, train_x, train_y) 
    