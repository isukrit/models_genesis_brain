## Modules
import os
os.environ['PYTHONHASHSEED'] = '0'

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

import pickle
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, Dropout, Conv2D, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

def model(x_train, y_train, num_labels, units, dropout, num_lstm_layers, num_conv_filters, model_type, batch_size):

	cnn_inputs = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2], 1), name='rnn_inputs')

	cnn_layer = Conv2D(num_conv_filters, kernel_size = (1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
	cnn_out = cnn_layer(cnn_inputs)

	sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
	sq_layer_out = sq_layer(cnn_out)

	rnn_layer = LSTM(units, return_sequences=True, name='lstm') #return_state=True
	rnn_layer_output = rnn_layer(sq_layer_out)

	dropout_layer = Dropout(rate = dropout)
	dropout_layer_output = dropout_layer(rnn_layer_output)

	dense_layer = Dense(num_labels, activation = 'softmax')
	dense_layer_output = dense_layer(dropout_layer_output)

	model = Model(inputs=cnn_inputs, outputs=dense_layer_output)

	print (model.summary())

	return model

def get_data(choice_task, src_dir, chunk_size):

	data_path = os.path.join(src_dir, choice_task)

	"""
	1. Remove unnecessary files (.ds_store) + faulty files (e.g. NaN)
	2. Import
	3. Flatten
	4. Combine
	5. Normalize all
	"""
	tasks = ['LANGUAGE_RL/', 'LANGUAGE_LR/', 'MOTOR_LR/','MOTOR_RL/'] #

	if choice_task not in tasks:
	    print('Invalid input detected. Allowable options: cn-mci, mci-ad, cn-ad')
	    exit()

	subject_names_list = []
	all_matrices = []
	all_matrices_labels = []

	for i, file_or_dir in enumerate(os.listdir(data_path)):
		matrix = np.load(data_path + file_or_dir)
		matrix = np.nan_to_num(matrix)
		num_time_points = np.array(matrix).shape[1]
		labels = matrix[-1]

		'''
		if matrix.shape[1] == 200:
			matrix = matrix[:, 10:150]
		elif matrix.shape[1] < 140:
			print (file_or_dir)
		'''
		chunk_num = 0
		for start, end in zip(range(0, num_time_points, chunk_size), range(chunk_size, num_time_points, chunk_size)):
			chunk_labels = labels[start:end]
			chunk_matrix = matrix[0:-1, start:end]
			# print (matrix.shape, np.min(matrix, axis=1).shape, np.max(matrix, axis=1).shape)
			all_matrices.append(chunk_matrix.T)
			all_matrices_labels.append(chunk_labels)
			subject_names_list.append(file_or_dir[-15:-6] ) #+ '_' + str(chunk_num)
			chunk_num += 1


	num_labels = np.max(all_matrices_labels) + 1
	num_samples = np.array(all_matrices).shape[0]

	'''
		Y_one_hot = np.zeros((num_samples, num_labels))

		for i, labels in enumerate(all_matrices_labels):
			for time_point in range(num_time_points):
				Y_one_hot[]
				Y[i, all_labels[i]] = 1  # 1-hot vectors
	'''

	return (subject_names_list, np.array(all_matrices), np.array(all_matrices_labels))

def convert_to_one_hot(y_labels, num_labels):
	num_samples = np.array(y_labels).shape[0]
	num_time_points = np.array(y_labels).shape[1]

	y_one_hot_labels = np.zeros((num_samples, num_time_points, num_labels))

	for i_sample, sample_time_points in enumerate(y_labels):
		for i_time, time_point in enumerate(sample_time_points):
			y_one_hot_labels[i_sample, i_time, int(y_labels[i_sample, i_time])] = 1

	return y_one_hot_labels

## Constants 
CHUNK_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]
EPOCH = 200
BATCH_SIZE = 32
LSTM_UNITS = 512 # https://zhuanlan.zhihu.com/p/58854907
CNN_FILTERS = 32
DROPOUT = 0.0
NUM_LSTM_LAYERS = 1
LEARNING_RATE = 0.05 # 0.001
SAVE_DIR = './'
PATIENCE = 10
COOLDOWN = 5 

MODEL_TO_RUN = 'lstm_with_cnn' # gru lstm ; must be in small letters

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

for CHUNK in CHUNK_SIZES:
	results_ground_truth = []

	(subject_names_list, data, labels) = get_data('LANGUAGE_RL/', '/data/sukrit001/Codes/RNN_attention/data/', CHUNK) #/data/sukrit001/Codes/RNN_attention/data/ # ['LANGUAGE_RL/', 'LANGUAGE_LR/', 'MOTOR_LR/','MOTOR_RL/'] #
	NUM_LABELS = int(np.max(labels) + 1)

	print('**************CHUNK SIZE ', CHUNK, '***********************')

	for SEED in range(1):
		print('**************SEED ', SEED, '***********************')
		sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
		K.set_session(sess)

		# Prepare dataset
		num_labels = np.max(labels, axis = None) + 1

		print("Dataset Dimensions" + str(data.shape))

		subjects_train, subjects_test, x_train, x_test, y_train, y_test = train_test_split(subject_names_list, data, labels, test_size=0.15, random_state=SEED, shuffle = False)

		subjects_train, subjects_test, x_train, x_test, y_train, y_test = np.array(subjects_train), np.array(subjects_test), np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

		print ('Train Subjects', subjects_train.shape, 'Train data shape', x_train.shape, 'Train labels shape', y_train.shape)
		print ('Test Subjects', subjects_test.shape, 'Test data shape', x_test.shape,'Test labels shape', y_test.shape)
		print ('Number of classes', NUM_LABELS)

		y_train_one_hot, y_test_one_hot = convert_to_one_hot(y_train, NUM_LABELS), convert_to_one_hot(y_test, NUM_LABELS)

		scalers = {}
		for i in range(x_train.shape[2]):
		    scalers[i] = StandardScaler()#MinMaxScaler(feature_range=(0, 1))
		    x_train[:, :, i] = scalers[i].fit_transform(x_train[:, :, i])
		    x_test[:, :, i] = scalers[i].transform(x_test[:, :, i])

		def get_lr_metric(optimizer):
		    def lr(y_true, y_pred):
		        return optimizer.lr
		    return lr

		# Train model
		rnn_model = model(x_train = x_train, y_train = y_train_one_hot, num_labels = NUM_LABELS, units = LSTM_UNITS, \
			dropout = DROPOUT, num_lstm_layers = NUM_LSTM_LAYERS, num_conv_filters = CNN_FILTERS, model_type = MODEL_TO_RUN, batch_size = BATCH_SIZE)

		opt = optimizers.Adam(clipnorm=1.)#(lr=LEARNING_RATE, clipnorm=1., decay=LEARNING_RATE/EPOCH)
		lr_metric = get_lr_metric(opt)

		rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', lr_metric])

		model_filename = SAVE_DIR + '/best_model_lstm_with_cnn_seed_' + str(SEED) + '_chunk_' + str(CHUNK) + '.h5'
		callbacks = [ModelCheckpoint(filepath=model_filename, monitor = 'val_acc', save_weights_only=True, save_best_only=True), EarlyStopping(monitor='val_acc', patience=PATIENCE)]#, LearningRateScheduler()]

		x_train_ = np.expand_dims(x_train, axis = 3)
		x_test_ = np.expand_dims(x_test, axis = 3)
		train_trailing_samples =  x_train_.shape[0]%BATCH_SIZE
		test_trailing_samples =  x_test_.shape[0]%BATCH_SIZE

		x_train_ = x_train_[0:-train_trailing_samples]
		x_test_ = x_test_[0:-test_trailing_samples]

		y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
		y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]

		history = rnn_model.fit(x_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(x_test_, y_test_one_hot))

		early_stopping_epoch = callbacks[1].stopped_epoch - PATIENCE + 1 # keras gives the 0-index value of the epoch, so +1
		print('Early stopping epoch: ' + str(early_stopping_epoch))

		if early_stopping_epoch < 0:
		    early_stopping_epoch = 100

		# Evaluate model and predict data on TEST 
		print("******Evaluating TEST set*********")

		scores_test = rnn_model.evaluate(x_test_, y_test_one_hot)
		print("test model: \n%s: %.2f%%" % (rnn_model.metrics_names[1], scores_test[1]*100))

		scores_train = rnn_model.evaluate(x_train_, y_train_one_hot)
		print("train model: \n%s: %.2f%%" % (rnn_model.metrics_names[1], scores_train[1]*100))
		
		y_test_predict = rnn_model.predict(x_test_)
		print(y_test_predict.shape)
		y_test_predict = np.argmax(y_test_predict, axis=2)
		y_test_predict = y_test_predict.flatten()
		# y_test_predict = convert_to_one_hot(y_test_predict, NUM_LABELS)

		y_test_ = np.array(y_test[0:-test_trailing_samples])

		print(y_test_predict.flatten().shape)
		print(y_test_.flatten().shape)

		cm = confusion_matrix(y_test_.flatten(), y_test_predict.flatten())
		print(cm)

		all_trainable_count = int(np.sum([K.count_params(p) for p in set(rnn_model.trainable_weights)]))

		MAE = metrics.mean_absolute_error(y_test_.flatten(), y_test_predict.flatten(), sample_weight=None, multioutput='uniform_average')

		'''
		try:
		    tn, fp, fn, tp = cm.ravel()
		    print(tn, fp, fn, tp)
		except ValueError:
		    print("100% accuracy, no CM to print")

		fpr_BDmodel, tpr_BDmodel, thresholds_BDmodel = roc_curve(y_test, y_test_predict)
		auc_BDmodel = auc(fpr_BDmodel, tpr_BDmodel)
			print("AUC: " + str(auc_BDmodel))
		'''
		with open(SAVE_DIR + '/results_test_' + MODEL_TO_RUN + '.csv', 'a') as out_stream:
		    out_stream.write(str(CHUNK) + ',' + str(SEED) + ',' + str(early_stopping_epoch) + ',' + str(all_trainable_count) + ',' + str(scores_test[1]*100) + ',' + str(scores_train[1]*100) + ',' + str(MAE) + '\n')

		K.clear_session()


		'''
		print(results_ground_truth)
		print("Average ground truth accuracy: " + str(statistics.mean(results_ground_truth)))
		print("Average ground stdev: " + str(statistics.stdev(results_ground_truth)))

		with open(SAVE_DIR + '/results_ground_truth' + MODEL_TO_RUN + '.csv', 'a') as out_stream:
		    out_stream.write(str(statistics.mean(results_ground_truth)) + ', ' + str(statistics.stdev(results_ground_truth)) + ', ' + str(results_ground_truth) + ', ' + str(early_stopping_epoch_list) + "\n")

		'''
