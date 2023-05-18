import tensorflow as tf
from tensorflow import keras
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(precision=4)

csv_file = "C:\\Users\\shion\\Desktop\\取引記録\\develop\\usdjpy.csv"
batch_size =64 
label = "date"
cols = ["date","open","close","max","min",'close_ave_last_5days','open_ave_last_5days','max_ave_last_5days','min_ave_last_5days','close_ave_last_20days','open_ave_last_20days','max_ave_last_20days','min_ave_last_20days']
epochs = 10
window =5 
df = pd.read_csv(
	csv_file,
	encoding="utf-8",
	index_col="date",
	parse_dates=True,
	usecols=['date','open','close','max','min']
)
df = df.sort_values('date')
train = df[:][: '2017']
test = df[:]['2018':]
print("train: {}".format(train.head()))
print("test: {}".format(test.head()))
#sc = MinMaxScaler()
#train_sc = pd.DataFrame(sc.fit_transform(train.values.reshape(-1,1)), index=train.index, columns=['scaled data'])
#test_sc = pd.DataFrame(sc.fit_transform(test.values.reshape(-1,1)), index=test.index, columns=['scaled data'])
#print("train_sc: {}".format(train_sc.head()))
#print("test_sc: {}".format(test_sc.head()))


#for i in range(1,window+1):
	#train_sc['shift_{}'.format(i)] = train_sc['scaled data'].shift(i)
	#test_sc['shift_{}'.format(i)] = test_sc['scaled data'].shift(i)
#train_dna = train_sc.dropna()
#test_dna = test_sc.dropna()
#train_y = train_dna[['scaled data']]
#train_x = train_dna.drop('scaled data',axis=1)
#test_y = test_dna[['scaled data']]
#test_x = test_dna.drop('scaled data',axis=1)
#input_dim = 1
#output_dim = 1
#hidden_units = 128
#model = keras.Sequential()
#model.add(keras.layers.LSTM(hidden_units,batch_input_shape=(None,window,input_dim),return_sequences=False))
#model.add(keras.layers.Dense(output_dim))
#model.compile(loss="mean_squared_error",optimizer='adam')
#model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_split=0.1)

#predicted = model.predict(test_x)
#d = pd.DataFrame(predicted[:],index=test_x.index)
#print(d)
#d.columns = ['predicted']
#d['test_y'] = test_y[:]
#ax = d.plot()
#test_y.plot(ax=ax)
#ax.legend(['predicted','test_y'])
#print(test_y)
#print(test_y.shape)
#plt.show()

