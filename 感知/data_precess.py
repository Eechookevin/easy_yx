import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras import models
from keras import layers

#从excel中读数据f
data = pd.read_excel(r'C:\Users\8aceMak1r\Desktop\data2.xlsx')
data.head()


#厂商、行业是离散数据使用独热编码
one_hot=OneHotEncoder()
data_temp=pd.DataFrame(one_hot.fit_transform(data[['厂商','行业','成本紧迫度需求']]).toarray(),
             columns=one_hot.get_feature_names(['厂商','行业','成本紧迫度需求']),dtype='int32')
data_onehot=pd.concat((data,data_temp),axis=1)    #也可以用merge,join


#影片数量是连续数据使用归一化操作
movie_num_datatmp = data['影片数量'].to_frame() #读数据并且增加一个维度
scaler = MinMaxScaler()
scaler = scaler.fit(movie_num_datatmp)
scaler.transform(movie_num_datatmp) #通过接口导出结果
movie_num = scaler.transform(movie_num_datatmp)


#由于此时的data_onehot的shape是（18，12）我们只取5列且转化为ndarry数据格式
dispersed_data = data_onehot.iloc[:,4:9].values

#标签是最后三列
label = data_onehot.iloc[:,9:].values

#合并两个离散矩阵和归一化矩阵作为网络的输入
net_input = np.hstack((dispersed_data,movie_num))


x_val = net_input[:4]
partial_x_train = net_input[4:]
y_val = label[:4]
partial_y_train = label[4:]
#=================================================================
#开始写网络
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(net_input.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

#模型编译
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(partial_x_train,
 partial_y_train,
 epochs=8,
 batch_size=4,
 validation_data=(x_val, y_val))



pass