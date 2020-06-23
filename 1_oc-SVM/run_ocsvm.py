#Copyright (c) 2019 ETH Zurich
#Author: Antonio Libri
#Licensed under the Apache License, Version 2.0 (see LICENSE file, on the main directory)
import pickle
import time
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import imp

### Set parameters
PATH="../dataset/dataset1/" # dataset path
#PATH="../dataset/dataset2/" # dataset path
#PATH="../dataset/dataset3/" # dataset path
PATH_SCALER_STORE="./models/" # path to store the scaler models 
PATH_SCALER_LOAD="./models/best_models/pwelch/dataset1/tst_outlTH_0.3_poly0.1_PCA_25/" # path where to load the scaler models 
#PATH_SCALER_LOAD="./models/best_models/examon/dataset1/tst_outlTH_0.3_poly0.1_PCA_25/" # path where to load the scaler models 
PATH_PCA_STORE="./models/" # path to store the PCA models 
PATH_PCA_LOAD="./models/best_models/pwelch/dataset1/tst_outlTH_0.3_poly0.1_PCA_25/" # path where to load the PCA models 
#PATH_PCA_LOAD="./models/best_models/examon/dataset1/tst_outlTH_0.3_poly0.1_PCA_25/" # path where to load the PCA models 
PATH_CLF_STORE="./models/" # path to store the oc-SVM models 
PATH_CLF_LOAD="./models/best_models/pwelch/dataset1/tst_outlTH_0.3_poly0.1_PCA_25/" # path where to load the oc-SVM models 
#PATH_CLF_LOAD="./models/best_models/examon/dataset1/tst_outlTH_0.3_poly0.1_PCA_25/" # path where to load the oc-SVM models 
METRIC=0 # Which metric to use for the analysis, namely 0=pwelch, 1=examon, 2=all
TRAINING=0 # Run training or Inference phase, namely 1=Training, 0=Inference
SCALER=2 # Which scaler to use, namely 0=None, 1=MinMax; 2=StandardScaler
PCA=1 # Run or not the PCA, namely 1=run_PCA, 0=do_not_run_PCA
HEALTHY_BENCH_WEIGHT=16 # used to weight the healthy predictions w.r.t. anomaly predictions, to compute a weighted F1-score, which takes into account the imbalance of the dataset (e.g., if we have 6 healthy acquisitions of a benchmark and 95 malware, we can set HEALTHY_BENCH_WEIGHT=16 to weight the healthy benchmark predictions vs. the malware predictions with an almost 50:50 rate)
OUTLIER_TH=0.3 # Threshold for percentage of outliers in the acquisition files (e.g., if greater than 0.3, we have a malware)
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
print('outlier_TH: '+str(OUTLIER_TH))

if TRAINING==1: # Training
  print('Running Training')
else: # Inference
  print('Running Inference')

### Initialize variables
filelist = imp.load_source('filelist', '../filelist.py')
files = filelist.files
bench_list = filelist.bench_list
START_WELCH=250 # Index where PSD Welch start
N_BENCH_IN_DATASET=7 # number of healthy benchmarks in the dataset folder
N_BENCH_ACQ=30 # total number of acquisitions (files) per benchmark in the dataset folder
N_MALW_ACQ=2  # total number of acquisitions (files) per benchmark per malware in the dataset folder
i_tMalw=files.index('v000_b00t1_idle_')

### Use TH-based method to predict malware vs. healthy benchmark
def check_sample(y,ok_TH):
  outliers = float(y[y == -1].size) # (-1=outlier)
  length = float(y.size)
  if (outliers/length)>ok_TH:
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

print('# files in dataset', len(x))
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

  ### Test dataset w. OC-SVM
  if TRAINING==1: # Training
    ### kernel = "linear", "poly", "rbf", "sigmoid"
    clf = svm.OneClassSVM(nu=0.1, kernel="poly", gamma=0.1)
    clf.fit(x_train)
    joblib.dump(clf, PATH_CLF_STORE+"clf"+str(iB)+".dump")

    y_test = {}
    y_pred = np.empty(0)
    y_true = np.empty(0)

    for k in range(len(x_test)):
      y_test[k] = clf.predict(x_test[k])

    for z in range(0,N_VAL_FILES): #Validation set (benchmarks)
      y_pred=np.append(y_pred,check_sample(y_test[z],OUTLIER_TH))
      y_true = np.append(y_true,[0])

    for z in range(0,N_MALW_TO_TEST): #Validation set (malware)
      y_pred=np.append(y_pred,check_sample(y_test[N_VAL_FILES+z],OUTLIER_TH))
      y_true = np.append(y_true,[1])

  else: # Inference
    clf = joblib.load(PATH_CLF_LOAD+"clf"+str(iB)+".dump")

    y_test = {}
    y_pred = np.empty(0)
    y_true = np.empty(0)
  
    for k in range(len(x_test)):
      y_test[k] = clf.predict(x_test[k])

    for z in range(0,N_TST_FILES): #Test set (benchmarks)
      y_pred=np.append(y_pred,check_sample(y_test[z],OUTLIER_TH))
      y_true = np.append(y_true,[0])

    for z in range(0,N_MALW_TO_TEST): #Test set (malware)
      y_pred=np.append(y_pred,check_sample(y_test[N_TST_FILES+z],OUTLIER_TH))
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
