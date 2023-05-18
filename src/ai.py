import tensorflow as tf
from tensorflow import keras
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(precision=4)
def create_model():
	#csv_file = tf.keras.utils.get_file("usdjpy.csv", "C:\\Users\\shion\\Desktop\\取引記録\\develop\\usdjpy.csv")
	#csv_file = "C:\\Users\\shion\\Desktop\\取引記録\\develop\\theDayBeforeGap.csv"
	csv_file = "C:\\Users\\shion\\Desktop\\trade\\develop\\simurater\\test.csv"
	batch_size = 1028 
	label = "LOCAL_TIME"
	epochs = 10
	window = 3 
	#df = pd.read_csv(csv_file)
	#print(df.head(10))
	#batches = tf.data.experimental.make_csv_dataset(
		#csv_file,
		#batch_size=batch_size,
		#label_name=label,
		#select_columns=cols,
		#num_epochs=epochs,
		#shuffle=False,

	#)
	#for b, l in batches.take(1):
		#print("label: {}".format(l))
		#print("features:")
		#for k, v in b.items():
			#print("{!r:20s}: {}".format(k,v))

	df = pd.read_csv(
		csv_file,
		encoding="utf-8",
		index_col="LOCAL_TIME",
		parse_dates =True,
		usecols=['LOCAL_TIME','CLOSE']
	)
	df = df.sort_values('LOCAL_TIME')
	#usecols=cols
	train = df['CLOSE'][:'2022-01']
	df_test = pd.read_csv(
		'C:\\Users\\shion\\Desktop\\trade\\develop\\simurater\\data.csv',
		encoding="utf-8",
		index_col="Local time",
		parse_dates =True,
		usecols=['Local time','Close']
	)
	#df_test = df_test[df_test['Volume']>0]
	df_test = df_test.sort_values('Local time')
	test = df_test['Close'][:]
	sc = MinMaxScaler()
	train_sc = pd.DataFrame(sc.fit_transform(train.values.reshape(-1,1)), index=train.index, columns=['scaled data'])
	test_sc = pd.DataFrame(sc.fit_transform(test.values.reshape(-1,1)), index=test.index, columns=['scaled data'])

	for i in range(1,window+1):
		train_sc['shift_{}'.format(i)] = train_sc['scaled data'].shift(i)
		test_sc['shift_{}'.format(i)] = test_sc['scaled data'].shift(i)
	train_dna = train_sc.dropna()
	test_dna = test_sc.dropna()
	train_y = train_dna[['scaled data']]
	train_x = train_dna.drop('scaled data',axis=1)
	test_y = test_dna[['scaled data']]
	test_x = test_dna.drop('scaled data',axis=1)
	#column count 
	input_dim = 1
	output_dim = 1
	hidden_units = 128
	model = keras.Sequential()
	if os.path.exists('./saved_model/test_model'):
		model = keras.models.load_model('./saved_model/test_model')
	else:
		model = keras.Sequential()
		model.add(keras.layers.LSTM(hidden_units,batch_input_shape=(None,window,input_dim),return_sequences=False))
		model.add(keras.layers.Dense(output_dim))
		model.compile(loss="mean_squared_error",optimizer='adam')
		model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_split=0.1)
		model.save('./saved_model/test_model')
	predicted = model.predict(test_x)
	inversed_test_y = pd.DataFrame(sc.inverse_transform(test_y),index=test_x.index)
	inversed_predicted = sc.inverse_transform(predicted)
	d = pd.DataFrame(inversed_predicted[:],index=test_x.index)
	print(d)
	d.columns = ['predicted']
	d['test_y'] = inversed_test_y[:]
	ax = d.plot()
	inversed_test_y.plot(ax=ax)
	ax.legend(['predicted','test_y'])
	print(inversed_test_y)
	print(inversed_test_y.shape)
	plt.show()

create_model()

#df_test = pd.read_csv(
	#'C:\\Users\\shion\\Desktop\\trade\\develop\\simurater\\test_USDJPY07.02.2022-12.02.2022.csv',
	#encoding="utf-8",
	#index_col="Local time",
	#parse_dates =True,
	#usecols=['Local time','Close']
#)
#print(df.head())
#model = keras.models.load_model('./saved_model/test_model')
