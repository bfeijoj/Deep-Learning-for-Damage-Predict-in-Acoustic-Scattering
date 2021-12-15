import pandas as pd
import numpy as np
import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def training(inspecting_location, path, hidden_layers, patience = 150, test_size = 0.1, learning_rate = 0.0005, decay = 3e-5, batch_size = 16, epochs = 100000):

	x_data = r'df_u_z_prior_' + inspecting_location + r'.csv'
	y_data = r'df_defect_field_non_transformed_prior_' + inspecting_location + r'.csv'

	df_X = pd.read_csv(os.path.join(path, x_data))
	df_y = pd.read_csv(os.path.join(path, y_data))

	X = np.array(df_X).T
	y = np.array(df_y).T

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

	model = Sequential()

	model.add(Flatten())
	model.add(Dense(np.shape(X)[1]))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	for ii in range(0, hidden_layers[0]):

		model.add(Dense(hidden_layers[1][ii]))
		model.add(Activation(hidden_layers[2][ii]))
		model.add(BatchNormalization())

	model.add(Dense(np.shape(y)[1]))
	model.add(Activation('linear'))
	model.add(BatchNormalization())

	opt = tf.keras.optimizers.Adam(learning_rate = learning_rate, decay = decay)

	model.compile(loss = 'mse',
	              optimizer = opt,
	              metrics = ['mse'])

	early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience)

	history = model.fit(X_train, y_train,
	                    batch_size = batch_size,
	                    epochs = epochs,
	                    validation_data = (X_test, y_test),
	                    callbacks = [early_stop])

	model.save('zz_3.model')

	training_conv = np.log(history.history['mse'])
	validation_conv = np.log(history.history['val_mse'])

	plt.figure(figsize = (12, 8))
	plt.plot(training_conv, label = 'Training', color = 'b')
	plt.plot(validation_conv, label = 'Validation', color = 'r')
	plt.title('Mean Squared Error', size = 28)
	plt.xlabel('Epochs', size = 28)
	plt.ylabel('log(Loss)', size = 28)
	plt.tick_params(axis='x', labelsize=20)
	plt.tick_params(axis='y', labelsize=20)
	plt.legend(fontsize = 20)
	plt.grid(color = 'b', alpha = 0.5, linestyle = 'dashed', linewidth = 0.5)
	plt.show()

	return plt.show()
