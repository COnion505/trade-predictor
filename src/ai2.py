import tensorflow as tf
from tensorflow import keras
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(precision=4)

save_path = './saved_model/ai2'
csv_file = "C:\\Users\\shion\\Desktop\\trade\\develop\\simurater\\test.csv"
batch_size=1028
epochs=5
label='Local time'
input_dim = 1
output_dim = 1
hidden_units = 100
window =5 

df= pd.read_csv(
	csv_file,
	encoding='utf-8',
	index_col=label,
	parse_dates=True,	
)
df = df.sort_values(label)
#train = df['CLOSE']['2020-01':'2020-09']
train = df[:]['2020-01':'2020-09']
#print('train: {}'.format(train))
#test = df['CLOSE']['2020-10':'2020-12']
test = df[:]['2020-10':'2020-12']
#print('test: {}'.format(test))
sc = MinMaxScaler()
train_sc = pd.DataFrame(sc.fit_transform(train.values.reshape(-1,1)),index=train.index, columns=['scaled'])
#train_sc['shift1'] = train_sc['scaled'].shift(1)
#train_sc = train_sc.dropna()
test_sc = pd.DataFrame(sc.fit_transform(test.values),index=test.index)
#test_sc['shift1'] = test_sc['scaled'].shift(1)
#test_sc = test_sc.dropna()

#train_y = train_sc[['scaled']]
#train_x = train_sc.drop('scaled', axis=1)
#test_y = test_sc[['scaled']]
#test_x = test_sc.drop('scaled', axis=1)

train_x = train_sc[['scaled']]
train_y = pd.DataFrame(sc.fit_transform(train['Close'].values.reshape(-1,1)),index=train.index,columns=['Close'])
test_x = test_sc[['scaled']]
test_y = pd.DataFrame(sc.fit_transform(test['Close'].values.reshape(-1,1)),index=test.index,columns=['Close'])


model = keras.Sequential([
	tf.keras.layers.LSTM(units=hidden_units,input_shape=(window,input_dim)),
	tf.keras.layers.Dense(output_dim,activation='relu')
])
model.compile(loss='mean_squared_error',optimizer='RMSprop')
if os.path.exists(save_path):
	model = keras.models.load_model(save_path)
else:
	model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_split=0.1,shuffle=False)
	model.save(save_path)
print('start predect...')
predicted = model.predict(test_x)
print('done')
inversed_predicted = pd.DataFrame(sc.inverse_transform(predicted),index=test_y.index)
inversed_test_y = pd.DataFrame(sc.inverse_transform(test_y),index=test_y.index)
#print(inversed_predicted)
#print(inversed_test_y)
ax = inversed_predicted.plot()
inversed_test_y.plot(ax=ax)
plt.show()