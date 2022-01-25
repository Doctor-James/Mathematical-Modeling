import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
import sklearn.feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_excel("./2.xlsx")
df = pd.DataFrame(data)
df1 = df[df["c1"]==1]
df2 = df[df["c1"]==2]
df3 = df[df["c1"]==3]
exam_x1 = df1.loc[:,"a1":"B17"]
exam_x2 = df2.loc[:,"a1":"B17"]
exam_x3 = df3.loc[:,"a1":"B17"]
exam_y1 = df1.loc[:,"D"]
exam_y2 = df2.loc[:,"D"]
exam_y3 = df3.loc[:,"D"]
x_test = np.array([81.48957655,81.97445123,85.16608081,81.45251238,81.00281808,83.5955792,84.44443909,73.66173617,2,48,1,15,3,1,1,1971,5,25,4,9,60,50,40,0,0
])
# x1_train,x1_test,y1_train,y1_test = sklearn.model_selection.train_test_split(exam_x1,exam_y1,train_size=1.0)
# x2_train,x2_test,y2_train,y2_test = sklearn.model_selection.train_test_split(exam_x2,exam_y2,train_size=1)
# x3_train,x3_test,y3_train,y3_test = sklearn.model_selection.train_test_split(exam_x3,exam_y3,train_size=1)
x1_train = exam_x1
y1_train = exam_y1
x2_train = exam_x2
y2_train = exam_y2
x3_train = exam_x3
y3_train = exam_y3
# x_train = x_train.values.reshape(-1,1)
# y_train = y_train.values.reshape(-1,1)
# x_test = x_test.values.reshape(-1,1)
# y_test = y_test.values.reshape(-1,1)
x_test = x_test.reshape(-1,25)
# #LR回归
# LR_1 = LogisticRegression(penalty='l1',solver="liblinear",C=1,random_state=420)
# LR_1.fit(x1_train,np.ravel(y1_train))
# importance = np.abs(LR_1.coef_).flatten()
# # sfm = SelectFromModel(LR_1, threshold=0.3)
# # sfm.fit(x1_train, np.ravel(y1_train))
# # X1_transform = sfm.transform(x1_train)
# print(importance)

#SVM

#将一维数据转化为多项式数据，然后采用线性SVM
# polynomial_svm_clf = Pipeline([ ("poly_featutres", PolynomialFeatures(degree=3)),
#                                 ("scaler", StandardScaler()),
#                                 ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))])
# polynomial_svm_clf.fit( x1_train, np.ravel(y1_train) )
# print(polynomial_svm_clf.predict(x_test))

#rbf kernel
rbf_kernel_svm_clf = Pipeline([
                                ("scaler", StandardScaler()),
                                ("svm_clf", SVC(kernel="rbf", gamma=5, C=1))
                            ])
rbf_kernel_svm_clf.fit( x1_train, np.ravel(y1_train) )
print(rbf_kernel_svm_clf)
# print(rbf_kernel_svm_clf.predict(x_test))



# # 计算出的第三高阈值
# importance = np.sort(importance.flatten())
# idx_third = importance.argsort()[-3]
# threshold = importance[idx_third] + 0.01
#
# sfm = SelectFromModel(LR_, threshold=threshold)
# sfm.fit(x1_train, np.ravel(y1_train))
# X_transform = sfm.transform(x1_train)
# print(X_transform.shape)


# 参数c及特征选择
# fullx = []
# fsx = []
#
# C = np.arange(0.01,10.01,0.5)
# for i in C:
#     LR_ = LogisticRegression(penalty='l2',solver="liblinear",C=i,random_state=420)
#     fullx.append(sklearn.model_selection.cross_val_score(LR_,x_train,np.ravel(y_train),cv=10).mean())
#     x_embedded = sklearn.feature_selection.SelectFromModel(LR_,norm_order=1).fit(x_train,np.ravel(y_train))
#     x_embedded_train = x_embedded.transform(x_train)
#     score = sklearn.model_selection.cross_val_score(LR_,x_embedded_train,np.ravel(y_train),cv=10).mean()
#     fsx.append(score)
#     print(score,i)
#print('特征选择前：',max(fullx),C[fullx.index(max(fullx))])   # 0.9824358974358974 0.51
#print('特征选择后',max(fsx),C[fsx.index(max(fsx))])      # 0.9875 0.51


