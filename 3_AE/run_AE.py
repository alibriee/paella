#Copyright (c) 2019 ETH Zurich
#Author: Antonio Libri
#Licensed under the Apache License, Version 2.0 (see LICENSE file, on the main directory)
import pickle
import time
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import numpy as np
import sys, os
import imp
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

### Set parameters
PATH="../dataset/dataset1/" # dataset path
#PATH="../dataset/dataset2/" # dataset path
#PATH="../dataset/dataset3/" # dataset path
PATH_AE_MODELS_LOAD="./models/best_models/pwelch/dataset1/tst_recTH_0.91_outlTH_0.3/" # path where to load the AE models 
#PATH_AE_MODELS_LOAD="./models/best_models/examon/dataset1/tst_recTH_0.91_outlTH_0.3/" # path where to load the AE models 
PATH_SCALER_STORE="./models/" # path to store the scaler models 
PATH_SCALER_LOAD="./models/best_models/pwelch/dataset1/tst_recTH_0.91_outlTH_0.3/"  # path where to load the scaler models 
#PATH_SCALER_LOAD="./models/best_models/examon/dataset1/tst_recTH_0.91_outlTH_0.3/" # path where to load the scaler models 
PATH_PCA_STORE="./models/" # path to store the PCA models 
PATH_PCA_LOAD="./models/" # path where to load the PCA models 
METRIC=0 # Which metric to use for the analysis, namely 0=pwelch, 1=examon, 2=all
TRAINING=0 # Run training or Inference phase, namely 1=Training, 0=Inference
SCALER=2 # Which scaler to use, namely 0=None, 1=MinMax; 2=StandardScaler
PLOT_LOSS=0 # Plot "Training vs. Validation loss", namely  0=do_not_plot, 1=plot_loss
ERROR=1 # Which estimation error to use, namely 0=Root Mean Square Error (RMSE), 1=Mean Square Error (MSE), 2=Mean Absolute Error (MAE) 
PCA=0 # Run or not the Principal Component Analysis (PCA), namely 1=run_PCA, 0=do_not_run_PCA
HEALTHY_BENCH_WEIGHT=16 # used to weight the healthy predictions w.r.t. anomaly predictions, to compute a weighted F1-score, which takes into account the imbalance of the dataset (e.g., if we have 6 healthy acquisitions of a benchmark and 95 malware, we can set HEALTHY_BENCH_WEIGHT=16 to weight the healthy benchmark predictions vs. the malware predictions with an almost 50:50 rate)
RECONSTR_ERR_TH=0.91 # Threshold for reconstruction error, to label monitored metrics (i.e., examon metrics or PSDs) as outliers (e.g., if the RECONSTR_ERR_TH of a PSD is greater than 0.91, we label the PSD as an outlier)
OUTLIER_TH=0.3 # Threshold for percentage of outliers in the acquisition files, to label the acquisition as malware (e.g., if OUTLIER_TH is greater than 0.3, we have a malware)
EPOCH=5 # Epoch to use for AutoEncoder
BATCH=8 # Batch size to use for AutoEncoder
OPTIMIZER='adagrad' # Optimizer to use for AutoEncoder
LOSS='mean_squared_error' # Loss function to use for AutoEncoder; Another option could be 'mean_absolute_error'
N_MALW_TO_TEST=95 # number of malware to test (min=1; max depends from the dataset used, namely max_dataset1=95, max_dataset2=6, max_dataset3=6)
N_BENCH_TO_TEST=7 # number of benchmarks to test  (min=1, max=7, all benchmarks)
N_TRAIN_FILES=18 # number of healthy benchmark acquisitions to use for training
N_VAL_FILES=6 # number of healthy benchmark acquisitions to use for validation
N_TST_FILES=6 # number of healthy benchmark acquisitions to use for test

# NOTE: by using N_TRAIN_FILES=18, N_VAL_FILES=6, N_TST_FILES=6, we splitted the dataset 
#       with 60% of the samples for Training, 20% for validation and 20% for test 


### Print Setup
if METRIC==0:
  print('\nInput Data: Power @20us')
elif METRIC==1:
  print('\nInput Data: Examon @20ms')
else:
  print('\nInput Data: Examon @20ms + Power @20us')

print("Dataset: "+PATH)
print('Reconstr_Err_TH: '+str(RECONSTR_ERR_TH))
print('outlier_TH: '+str(OUTLIER_TH))
print('hidden_layers_dim: '+str(8)+'.'+str(4)+'.'+str(4))
print('loss: '+LOSS)
print('optimizer: '+OPTIMIZER)
print('epoch: '+str(EPOCH))
print('batch: '+str(BATCH))

### Initialize variables
filelist = imp.load_source('filelist', '../filelist.py')
files = filelist.files
bench_list = filelist.bench_list
START_WELCH=250 # Index where PSD Welch start
N_BENCH_IN_DATASET=7 # number of healthy benchmarks in the dataset folder
N_BENCH_ACQ=30 # total number of acquisitions (files) per benchmark in the dataset folder
N_MALW_ACQ=2  # total number of acquisitions (files) per benchmark per malware in the dataset folder
i_tMalw=files.index('v000_b00t1_idle_')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To disable TF messages
tf.logging.set_verbosity(tf.logging.ERROR) # Enable only Error TF messages

### Use TH-based method to predict malware vs. healthy benchmark
def check_sample(x,y,reconstrErr_TH,outlier_TH,error):
  if error==0: # RMSE (Root Mean Squared Error)
    reconstr_error_array = np.sqrt(np.mean(np.power(x - y, 2), axis=1))
  elif error==1: # MSE (Mean Squared Error)
    reconstr_error_array = np.mean(np.power(x - y, 2), axis=1)
  else: # MAE (Mean Absolut Error)
    reconstr_error_array = np.mean(np.abs(x - y), axis=1)

  outliers = float(reconstr_error_array[reconstr_error_array>reconstrErr_TH].size)
  length = float(reconstr_error_array.size)

  if (outliers/length)>outlier_TH:
    res=1 # Predicted as malware (TP or FP)
  else:
    res=0 # Predicted as not malware (TN or FN)
  return res

### Load dataset from pickle
starttime = time.time()
x = {}

for i in range(0,N_BENCH_IN_DATASET*N_BENCH_ACQ+N_MALW_TO_TEST*N_MALW_ACQ*N_BENCH_IN_DATASET):
  infile = open(PATH+files[i],'rb')
  x[i] = pickle.load(infile)
  infile.close()

print('# files in dataset: '+str(len(x)))
print('Loading dataset took {} seconds\n'.format(round((time.time() - starttime),2)))

starttime = time.time()
y_pred_TOT = np.empty(0)
y_true_TOT = np.empty(0)

# iB from 0-6 --> 0=idle, 1=qe, 2=hpl, 3=hpcg, 4=grom, 5=btC9, 6=btC16
for iB in range(0,N_BENCH_TO_TEST):
  if TRAINING==1: # Training
    x_train = x[N_BENCH_ACQ*iB+0]
    for z in range(1,N_TRAIN_FILES):
      x_train = np.concatenate((x_train,x[N_BENCH_ACQ*iB+z]),axis=0) # Training set

    if METRIC==0:
      x_train = x_train[:,START_WELCH:] #pwelch only
    elif METRIC==1:
      x_train = x_train[:,:START_WELCH] #examon only
    #else: --> all
  
    ### Standardize the Data
    if(SCALER==1):
      from sklearn.preprocessing import MinMaxScaler
      scaler = MinMaxScaler()
    elif(SCALER==2):
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
  
    if(SCALER==1 or SCALER==2):
      scaler.fit(x_train)
      x_train = scaler.transform(x_train)
      joblib.dump(scaler, PATH_SCALER_STORE+"scaler"+str(iB)+".dump")

  else: # Inference
    scaler = joblib.load(PATH_SCALER_LOAD+"scaler"+str(iB)+".dump")

  x_test = {}

  # ** benchmarks
  if TRAINING==1: #Training (Validation Set)
    start_bench=0
    end_bench=N_VAL_FILES
  else: # Inference (Test Set)
    start_bench=N_VAL_FILES
    end_bench=N_TST_FILES

  for z in range(0,end_bench): # Validation + Test Set (Bench)
    x_test[z] = x[N_BENCH_ACQ*iB+N_TRAIN_FILES+start_bench+z]

    if METRIC==0:
      x_test[z] = x_test[z][:,START_WELCH:] #pwelch only
    elif METRIC==1:
      x_test[z] = x_test[z][:,:START_WELCH] #examon only
    #else: --> all

    if(SCALER==1 or SCALER==2):
      x_test[z] = scaler.transform(x_test[z])

  # ** malware
  if TRAINING==1: #Training (Validation Set)
    start_malwr=0
  else: # Inference (Test Set)
    start_malwr=1

  for malw in range(0,N_MALW_TO_TEST): # Validation OR Testset
    x_test[end_bench+malw] = x[i_tMalw+malw*N_BENCH_IN_DATASET*N_MALW_ACQ+iB*N_MALW_ACQ+start_malwr]

    if METRIC==0:
      x_test[end_bench+malw] = x_test[end_bench+malw][:,START_WELCH:] #pwelch only
    elif METRIC==1:
      x_test[end_bench+malw] = x_test[end_bench+malw][:,:START_WELCH] #examon only
    #else: --> all

    if(SCALER==1 or SCALER==2):
      x_test[end_bench+malw] = scaler.transform(x_test[end_bench+malw])

  if(PCA==1):
    if TRAINING==1: # Training
      from sklearn.decomposition import PCA
      pca = PCA(n_components=25)

      pca.fit(x_train)
      x_train = pca.transform(x_train)
      joblib.dump(pca, PATH_PCA_STORE+"pca"+str(iB)+".dump")
    else: # Inference
      pca = joblib.load(PATH_PCA_LOAD+"pca"+str(iB)+".dump")    

    for k in range(len(x_test)):
      x_test[k] = pca.transform(x_test[k])

  ### Test dataset w. AutoEncoder
  if TRAINING==1: #Training
    print('Training '+bench_list[iB])
    n_features = x_train.shape[1]

    input_data = Input(shape=(n_features, ))

    encoder = Dense(8, activation='tanh',
                    activity_regularizer=regularizers.l1(10e-5))(input_data)
    encoder = Dense(4, activation='relu')(encoder)
    decoder = Dense(4, activation='relu')(encoder)
    decoder = Dense(n_features, activation='tanh')(decoder)
    autoencoder = Model(inputs=input_data, outputs=decoder)
    autoencoder.compile(optimizer=OPTIMIZER, 
                        loss=LOSS, 
                        metrics=['mse'])
  
    checkpointer = ModelCheckpoint(filepath="./models/model_Bench"+str(iB)+".h5",
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    history = autoencoder.fit(x_train, x_train,
                        epochs=EPOCH,
                        batch_size=BATCH,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=0,
                        callbacks=[checkpointer, tensorboard]).history

    ### Plot loss per benchmark
    if PLOT_LOSS==1:
      plt.figure()
      plt.plot(history['loss'])
      plt.plot(history['val_loss'])
      plt.title('Training vs. validation loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['Training loss', 'Validation loss'], loc='upper right')
      plt.draw()

    y_test = {}
    y_pred = np.empty(0)
    y_true = np.empty(0)

    for k in range(len(x_test)):
      y_test[k] = autoencoder.predict(x_test[k])

    for z in range(0,N_VAL_FILES): # Test set (benchmarks)
      y_pred=np.append(y_pred,check_sample(x_test[z],y_test[z],RECONSTR_ERR_TH,
                       OUTLIER_TH,ERROR))
      y_true = np.append(y_true,[0])

    for z in range(0,N_MALW_TO_TEST): #Test set (malware)
      y_pred=np.append(y_pred,check_sample(x_test[N_VAL_FILES+z],y_test[N_VAL_FILES+z],
                       RECONSTR_ERR_TH, OUTLIER_TH,ERROR))
      y_true = np.append(y_true,[1])

  else: # Inference
    print("loading pre-stored model "+bench_list[iB])
    autoencoder = load_model(PATH_AE_MODELS_LOAD+"model_Bench"+str(iB)+".h5")

    y_test = {}
    y_pred = np.empty(0)
    y_true = np.empty(0)
  
    for k in range(len(x_test)):
      y_test[k] = autoencoder.predict(x_test[k])
  
    for z in range(0,N_TST_FILES): # Test set (benchmarks)
      y_pred=np.append(y_pred,check_sample(x_test[z],y_test[z],RECONSTR_ERR_TH,
                       OUTLIER_TH,ERROR))
      y_true = np.append(y_true,[0])

    for z in range(0,N_MALW_TO_TEST): #Test set (malware)
      y_pred=np.append(y_pred,check_sample(x_test[N_TST_FILES+z],y_test[N_TST_FILES+z],
                       RECONSTR_ERR_TH,OUTLIER_TH,ERROR))
      y_true = np.append(y_true,[1])
  
  y_pred_TOT=np.append(y_pred_TOT,y_pred)
  y_true_TOT=np.append(y_true_TOT,y_true)

  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  fpr=float(fp)/(fp+tn)
  fnr=float(fn)/(fn+tp)
  f1_score_w=float(2*tp)/(2*tp+fp*HEALTHY_BENCH_WEIGHT+fn)
  print('Benchmark: '+bench_list[iB])
  print('y_true.size: %i' % y_true.size)
  print('y_pred.size: %i' % y_pred.size)
  print('TN: %i, FP: %i, FN: %i, TP: %i' % (tn, fp, fn, tp))
  print('FPR: %f, FNR: %f' % (fpr,fnr))
  print('F1-score weighted: %f' % (f1_score_w))
  print('')

if PLOT_LOSS==1:
  plt.show()

# Print Final Stats
print('Statistics for entire dataset:')
tn, fp, fn, tp = confusion_matrix(y_true_TOT, y_pred_TOT).ravel()
fpr=float(fp)/(fp+tn)
fnr=float(fn)/(fn+tp)
f1_score_w=float(2*tp)/(2*tp+fp*HEALTHY_BENCH_WEIGHT+fn)
print('TN: %i, FP: %i, FN: %i, TP: %i' % (tn, fp, fn, tp)) 
print('FPR: %f, FNR: %f' % (fpr,fnr))
print('F1-score weighted: %f' % (f1_score_w))
print('All code took {} seconds'.format(round((time.time() - starttime),2)))
print('done!')
