import numpy as np
import pandas as pd 
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from scipy import stats
from scipy.stats import norm
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import damageField as damf
from acousticScattering import acousticScattering

def predict(noise_level, damage_location, damage_field):

	h_ref = 1e-2
	scale_factor = 10e-2 / h_ref
	x = np.arange(- 50 * scale_factor, 50 * scale_factor, scale_factor / 7)
	dx = x[1] - x[0]
	lx = len(x)

	if damage_location == 'zz_3':

		(u_solution_x_fft, u_solution_z_fft, u_solution_x, u_solution_z) = acousticScattering(inspecting_location = 'zz_3',
																					     damage_distribution_zz_3 = damage_field)

	elif damage_location == 'zz_2':

		(u_solution_x_fft, u_solution_z_fft, u_solution_x, u_solution_z) = acousticScattering(inspecting_location = 'zz_2',
																					     damage_distribution_zz_2 = damage_field)

	elif damage_location == 'xx_3':

		(u_solution_x_fft, u_solution_z_fft, u_solution_x, u_solution_z) = acousticScattering(inspecting_location = 'xx_3',
																					     damage_distribution_xx_3 = damage_field)

	elif damage_location == 'xx_2':

		(u_solution_x_fft, u_solution_z_fft, u_solution_x, u_solution_z) = acousticScattering(inspecting_location = 'xx_2',
																					     damage_distribution_xx_2 = damage_field)

	else:

		return print('invalid location')


	model_name = damage_location+'.model'
	model = tf.keras.models.load_model(model_name)
	X_sample = u_solution_z.reshape(1, 700)
	max_X_value = np.max(X_sample)

	X_predict = np.zeros(X_sample.shape)

	for ii in range(0, X_sample.shape[1]):
	    
	    X_predict[0, ii] = X_sample[0, ii] + (noise_level * max_X_value * np.random.randn())

	    
	prediction = model.predict(X_predict)

	field_transformed = np.tanh(np.exp(-prediction))
	field_transformed = field_transformed - max(field_transformed.T)

	filter_origin = 0
	filter_sigma = 2
	filter_offset = x[-1] - 50
	field_filter = np.zeros(x.shape)

	for ii in range(0, lx):
	    field_filter[ii] = np.exp(- np.max([0, abs(x[ii] - filter_origin) - filter_offset]) / (2 * filter_sigma ** 2))

	final_field = field_filter * field_transformed + 1

	final_field = final_field.reshape(700,)

	return final_field