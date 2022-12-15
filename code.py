# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:20:38 2022

@author: My
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data=np.loadtxt(open('data.csv',encoding='GB2312'),dtype=np.str,delimiter=',',skiprows=2)
data_=data[:,1:].astype('float')
x=np.zeros((50,7)) #净利润，期末总股本，平均净资产，期末现金流净额，本期主营收入，上期主营收入，上期净利润，流通股的绝对值，期末净资产
x[:,0]=data_[:,0]/data_[:,1] #每股收益a
x[:,1]=data_[:,0]/data_[:,2] #净资产收益率b
x[:,2]=data_[:,3]/data_[:,1] #每股经营现金流量净额c
x[:,3]=data_[:,4]/data_[:,5]-1#主营业务收入增长率d
x[:,4]=data_[:,0]/data_[:,6]-1#净利润增长率e
x[:,5]=np.absolute(data_[:,7]) #流通股本f
x[:,6]=data_[:,8]/data_[:,1] #每股净资产g

#机器学习标准化
scaler=StandardScaler()
X_after=scaler.fit_transform(x)  #2021年标准化数据
np.savetxt('data1.csv',X_after,delimiter=',') 

data2019=np.loadtxt(open('data2019.csv',encoding='GB2312'),dtype=np.str,delimiter=',',skiprows=1)
data2019_=data2019[:,1:10].astype('float')
x_=np.zeros((50,7)) #净利润，期末总股本，平均净资产，期末现金流净额，本期主营收入，上期主营收入，上期净利润，流通股的绝对值，期末净资产
x_[:,0]=data2019_[:,0]/data2019_[:,1] #每股收益a
x_[:,1]=data2019_[:,0]/data2019_[:,2] #净资产收益率b
x_[:,2]=data2019_[:,3]/data2019_[:,1] #每股经营现金流量净额c
x_[:,3]=data2019_[:,4]/data2019_[:,5]-1#主营业务收入增长率d
x_[:,4]=data2019_[:,0]/data2019_[:,6]-1#净利润增长率e
x_[:,5]=np.absolute(data2019_[:,7]) #流通股本f
x_[:,6]=data2019_[:,8]/data2019_[:,1] #每股净资产g

#机器学习标准化
scaler=StandardScaler()
X_before=scaler.fit_transform(x_)  #2019年标准化数据
np.savetxt('data2019norm.csv',X_before,delimiter=',')




cluster_data=np.loadtxt('results.csv',skiprows=1,delimiter=',')
class_=cluster_data[:,7] #聚类类别
#softmax建模(5步)1.确定activation function 2.确定损失函数 3.拟合模型，使得cost最小 4.获取w,b系数 5.预测。
#注意tensorflow建模时数据保证为2-D,与传统的numpy不同。
#%%
model=Sequential(
    [
     Dense(units=3,activation='relu',name='L1'),
     Dense(units=3,activation='linear',name='L2')
     ]
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),#此处是优化gradient descent
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)) #output函数运用的是linear,在这里调用logits

model.fit(X_after,class_,epochs=100)
#%%
w1,b1=model.get_layer('L1').get_weights()
w2,b2=model.get_layer('L2').get_weights()
print(w1,b1)  #第一列神经元代表成长能力。第二列神经元代表股本扩张能力，第三列代表盈利能力
print(w2,b2)
#prediction
logit=model(X_after) #获得z值
predictions=tf.nn.softmax(logit).numpy()
#%%
m=predictions.shape[0]
prediction=np.zeros((50,))
for i in range(m):
    prediction[i]=np.argmax(predictions[i,:])+1


#精确度
n=0
print('不匹配的股票有')
for i in range(m):
    if class_[i]==prediction[i]:
        n+=1
    else:
        print(f'聚类下的种类为{class_[i]},神经网络下的种类为{prediction[i]},公司为：{data[i,0]},索引为{i}')

print(f'精确度为{n/m}')

u=0
p=0
o=0
for i in range(len(prediction)):
    if prediction[i]==1:
        u+=1
    elif prediction[i]==2:
        p+=1
    else:
        o+=1
print(f'第一类有{u},第二类有{p},第三类有{o}')

classes=np.unique(class_)

value=np.zeros((3,7))
standard=np.zeros((3,7))
for i in range(len(classes)):
    for j in range(X_after.shape[1]):
        ind=np.where(prediction==i+1)     
        value[i,j]=X_after[ind,j].mean()
        standard[i,j]=X_after[ind,j].std()
        

#%%

#分类图
z=np.zeros((50,3))
for i in range(X_after.shape[0]):
    z[i]=np.matmul(X_after[i],w1)+b1#这是三个神经元的值
    

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
classes=np.unique(prediction)
colors=['blue','red','yellow']
labels=['first class','second class','third class']
for j in range(len(classes)):
    ind=np.where(prediction==j+1)#选择的分类对象
    ax.scatter(z[ind,0],z[ind,1],z[ind,2],label=labels[j])
ax.legend(loc='upper left')
ax.set_xlabel('profitability',fontsize=10)
ax.set_ylabel('Equity expansion capacity',fontsize=10)
ax.set_zlabel('Ability of growth',fontsize=10)
ax.set(xlim=(-6,4),ylim=(-4,6),zlim=(-4,4))
ax.set_title('2021s')
#第一列为盈利能力，第二列为股本扩张能力，第三列为成长能力对于z

#标记
def circle_(i,a,b,r):
    circle=np.arange(0,2*np.pi,0.01)
    x=a+r*np.cos(circle)
    y=b+r*np.sin(circle)
    return ax[i].plot(x,y,c='black')
#二维更清楚
fig,ax=plt.subplots(1,3,figsize=(10,4))
for j in range(len(classes)):
    ind=np.where(prediction==j+1)
    ax[0].scatter(z[ind,0],z[ind,1],label=labels[j],color=colors[j])#不同分组中，盈利能力和股本扩张能力关系
    ax[0].set_xlabel('profitability')
    ax[0].set_ylabel('Equity expansion capacity')
    ax[0].set(xlim=(-7.5,2.5),ylim=(-4,6))
    #circle_(0,-1,1.5,1)
    ax[1].scatter(z[ind,0],z[ind,2],label=labels[j],color=colors[j])#不同分组中，盈利能力和成长能力关系
    ax[1].set_xlabel('profitability')
    ax[1].set_ylabel('Ability of growth')
    ax[1].set(xlim=(-7.5,2.5),ylim=(-4,4))
    ax[2].scatter(z[ind,1],z[ind,2],label=labels[j],color=colors[j])#不同分组中，股本扩张能力和成长能力关系
    ax[2].set_xlabel('Equity expansion capacity')
    ax[2].set_ylabel('Ability of growth')
    ax[2].set(xlim=(-4,6),ylim=(-4,4))
fig.suptitle('the realtionships among the number of different classification in 2021s',fontsize=20)
fig.tight_layout(pad=0.2)
plt.legend()




#%%
cluster_data_=np.loadtxt('results1.csv',skiprows=1,delimiter=',')
class_1=cluster_data_[:,7] #聚类类别


model=Sequential(
    [
     Dense(units=3,activation='relu',name='L1'),
     Dense(units=3,activation='linear',name='L2')
     ]
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),#此处是优化gradient descent
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)) #output函数运用的是linear,在这里调用logits

model.fit(X_before,class_1,epochs=100)

#%%
w1_,b1_=model.get_layer('L1').get_weights()
w2_,b2_=model.get_layer('L2').get_weights()
print(w1_,b1_)  #第一列神经元代表成长能力。第二列神经元代表股本扩张能力，第三列代表盈利能力
print(w2_,b2_)
#prediction
logits=model(X_before) #获得z值
predictions_=tf.nn.softmax(logits).numpy()

#%%
m=predictions_.shape[0]
prediction_=np.zeros((50,))
for i in range(m):
    prediction_[i]=np.argmax(predictions_[i,:])+1

#精确度
n=0
print('不匹配的股票有')
for i in range(m):
    if class_1[i]==prediction_[i]:
        n+=1
    else:
        print(f'聚类下的种类为{class_1[i]},神经网络下的种类为{prediction_[i]},公司为：{data2019[i,0]},索引为{i}')

print(f'精确度为{n/m}')
q=0
r=0
t=0
for i in range(len(prediction_)):
    if prediction_[i]==1:
        q+=1
    elif prediction_[i]==2:
        r+=1
    else:
        t+=1
print(f'第一类有{q},第二类有{r},第三类有{t}')

value_=np.zeros((3,7))
standard_=np.zeros((3,7))
for i in range(len(classes)):
    for j in range(X_before.shape[1]):
        ind=np.where(prediction_==i+1)
        value_[i,j]=X_before[ind,j].mean()
        standard_[i,j]=X_before[ind,j].std()
        

#%%
#分类图
z_=np.zeros((50,3))
for i in range(X_before.shape[0]):
    z_[i]=np.matmul(X_before[i],w1_)+b1_#这是三个神经元的值
    

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
classes=np.unique(class_1)
colors=['blue','red','yellow']
labels=['first class','second class','third class']
for j in range(len(classes)):
    ind=np.where(prediction_==j+1)
    ax.scatter(z_[ind,1],z_[ind,0],z_[ind,2],label=labels[j]) #顺序为盈利能力，股本扩张能力，成长能力
ax.legend(loc='upper left')
ax.set_xlabel('profitability',fontsize=10)
ax.set_ylabel('Equity expansion capacity',fontsize=10)
ax.set_zlabel('Ability of growth',fontsize=10)
ax.set(xlim=(-6,4),ylim=(-4,6),zlim=(-4,4))
ax.set_title('2019s')
#第一列为股本扩张能力，第二列为盈利能力，第三列为成长能力对于z
    
#二维更清楚
fig,ax=plt.subplots(1,3,figsize=(10,4))
for j in range(len(classes)):
    ind=np.where(prediction_==j+1)
    ax[0].scatter(z_[ind,1],z_[ind,0],label=labels[j],color=colors[j])#不同分组中，盈利能力和股本扩张能力关系
    ax[0].set_xlabel('profitability')
    ax[0].set_ylabel('Equity expansion capacity')
    ax[0].set(xlim=(-7.5,2.5),ylim=(-4,6))
    ax[1].scatter(z_[ind,1],z_[ind,2],label=labels[j],color=colors[j])#不同分组中，盈利能力和成长能力关系
    ax[1].set_xlabel('profitability')
    ax[1].set_ylabel('Ability of growth')
    ax[1].set(xlim=(-7.5,2.5),ylim=(-4,4))
    ax[2].scatter(z_[ind,0],z_[ind,2],label=labels[j],color=colors[j])#不同分组中，股本扩张能力和成长能力关系
    ax[2].set_xlabel('Equity expansion capacity')
    ax[2].set_ylabel('Ability of growth')
    ax[2].set(xlim=(-4,6),ylim=(-4,4))
fig.suptitle('the realtionships among the number of different classification in 2019s',fontsize=20)
fig.tight_layout(pad=0.2)
plt.legend()

#%%
a = ['第一类','第二类','第三类']
b_14 = [6,35,9]
b_15 = [14,15,21]
barwidth = 0.2
x_14 = range(len(a))
x_15 = [i+barwidth for i in x_14]
x_16 = [i+barwidth*2 for i in x_14]
plt.figure(figsize=(20, 8), dpi=80)
plt.bar(x_14, b_14, width=0.2, label="2019s")
plt.bar(x_15, b_15, width=0.2, label="2021s")
_xtick_labels = a
plt.xticks(x_15, _xtick_labels)
plt.legend(loc="upper right")
plt.show()

