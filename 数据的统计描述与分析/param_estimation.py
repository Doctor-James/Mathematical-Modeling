import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import scipy.stats
#读取csv中数据并转换成list
def get_data(lines):
    sizeArry=[]
    for line in lines:
        line = line.replace("\n","") # 因为读出来的数据每一行都有一个回车符，要删除
        line = float(line)
        sizeArry.append(line)
    return sizeArry

#画直方图
def draw_hist(lenths):  #lenths 接受的其实是 sizeArry传来的数组 就是def get_data(lines) 返回的数据
    data = lenths
# 对数据进行切片，将数据按照从最小值到最大值分组，分成20组
    bins = np.linspace(min(data),max(data),20)
# 这个是调用画直方图的函数，意思是把数据按照从bins的分割来画
    plt.hist(data,bins)
#设置出横坐标
    plt.xlabel('Number of ×××')
#设置纵坐标的标题
    plt.ylabel('Number of occurences')
#设置整个图片的标题
    plt.title('Frequency distribution of number of ×××')

    plt.show()


def write_data():
    random.seed(100)
    X =[]
    for i in range(100):
        X.append(random.gauss(5,2))
    with open("./data.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(X)):
            writer.writerow([X[i]])


# 正态分布下的置信区间
def norm_conf (data,confidence=0.95):
    sample_mean = np.mean(data)
    sample_std = np.std(data,ddof=1) #无偏估计
    sample_size = len(data)
    conf_intveral = scipy.stats.norm.interval(confidence, loc=sample_mean, scale=sample_std/np.sqrt(sample_size))
    print(conf_intveral)

# T分布下的置信区间
def ttest_conf (data,confidence=0.95):
    sample_mean = np.mean(data)
    sample_std = np.std(data,ddof=1)
    sample_size = len(data)
    conf_intveral = scipy.stats.t.interval(confidence,df = (sample_size-1) , loc=sample_mean, scale=sample_std)
    print(conf_intveral)

if __name__ == "__main__":
    f = open("./data.csv")
    lines = f.readlines()
    X = get_data(lines)
    norm_conf(X)

