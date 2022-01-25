import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

#pd.set_option('mode.chained_assignment', None)
data = pd.read_excel("./1.xlsx")
data_write = data
#print(data.a1.describe())
#a异常值
#print(data[data['a5']>=100])
# carat_table = pd.crosstab(index=data["a5"], columns="count")
# print(carat_table)

#画散点图
# df = pd.DataFrame(data,columns=['a1','c1'])
# df.plot.scatter( x='c1',y='a1')
# plt.show()

#找b中异常
#print(data[data['B13']<data['B14']])
# print(data.loc[data['B13']<data['B14'],'c0'].values)

data_right = data[(data['B13']>data['B14']) & (data['B13']>data['B15'])]

#求有效数据中b15/b13的中位数
df_right = pd.DataFrame(data_right,columns=['B13','B14','B15'])
df_right['D0'] = df_right['B15']/df_right['B13']
df_right_va = df_right['D0'].values
df_right_vanp = np.array(df_right_va)
df_right_vanp = np.sort(df_right_vanp)
mul1 = df_right_vanp[int(df_right_vanp.shape[0]/2)]



data_wrong = data[data['B13']<data['B15']]
df_wrong = pd.DataFrame(data_wrong,columns=['B13','B15'])
df_wrong['B13'] = df_wrong['B15']/mul1
df = pd.DataFrame(data_write)

for index in df_wrong.index:
     df.loc[index,'B13'] =  df_wrong.loc[index,'B13']



data_right = data_write[(data_write['B13']>data_write['B14'])]
#求有效数据中b14/b13的中位数
df_right = pd.DataFrame(data_right,columns=['B13','B14','B15'])
df_right['D1'] = df_right['B14']/df_right['B13']
df_right_va = df_right['D1'].values
df_right_vanp = np.array(df_right_va)
df_right_vanp = np.sort(df_right_vanp)
mul2 = df_right_vanp[int(df_right_vanp.shape[0]/2)]

data_wrong = data_write[data_write['B13']<data_write['B14']]
df_wrong = pd.DataFrame(data_wrong,columns=['B13','B14','B15'])
df_wrong['B14'] = df_wrong['B13']*mul2

for index in df_wrong.index:
     df.loc[index,'B14'] =  df_wrong.loc[index,'B14']
df.fillna(0, inplace=True)
df.to_excel('2.xlsx', index=False)