Copyright (c) 2019 ETH Zurich  
Author: Antonio Libri  
Licensed under the Apache License, Version 2.0 (see LICENSE file, on the main directory)  

# pAElla (Power-AutoEncoder-weLch for anomaLy and Attacks): Edge-AI based Real-Time Malware Detection in Data Centers

In this repository, we share the implementation of our work called Paella. 
For details, please refer to the papers below.

If this code proves useful for your research, please cite:
> [1] Antonio Libri, Andrea Bartolini, and Luca Benini,
"pAElla: Edge-AI based Real-Time Malware Detection in Data Centers", 
IEEE Internet of Things Journal, 2020.
Available on [ieeexplore](https://ieeexplore.ieee.org/abstract/document/9060937)

and
> [2] Antonio Libri, Andrea Bartolini, and Luca Benini,
"Dig: Enabling out-of-band scalable high-resolution monitoring for data-center analytics, automation and control", 
The 2nd International Industry/University Workshop on Data-center Automation, Analytics, and Control (DAAC 2018), Dallas, Texas, USA, 2018. 
Available on [ETH Research Collection](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/306925/DIG_2018.pdf?sequence=5)

### Installing Dependencies

The code is written in python, and exploit Keras and Tensorflow as back-end for running our method with the AutoEncoder ([AE](https://www.tensorflow.org/guide/keras/functional)), while scikit-learn APIs for running it with one-class SVM ([oc-SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)) and Isolation Forest ([IF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)). If the machine where you run the code includes a GPU, we recommend to install TensorFlow for GPU. Further, we have used conda as a python package manager and exported the environment specifications to `conda-env-paella.yml`. 
You can recreate our environment by running 

```
$ conda env create -f conda-env-paella.yml -n paella 
```

Make sure to activate the environment before running any code.

### Download the dataset

Please, download all 6 compressed files containing the dataset from [1](https://www.dropbox.com/s/syzfelke5tmew8x/paella_dataset.tar.gz.part1?dl=0),[2](https://www.dropbox.com/s/rqmqg77m2l3er28/paella_dataset.tar.gz.part2?dl=0),[3](https://www.dropbox.com/s/paqplvtzamk8wyk/paella_dataset.tar.gz.part3?dl=0),[4](https://www.dropbox.com/s/uhbdjf1aejcmobc/paella_dataset.tar.gz.part4?dl=0),[5](https://www.dropbox.com/s/z9nynypt67dpp9r/paella_dataset.tar.gz.part5?dl=0),[6](https://www.dropbox.com/s/ugtwbf4603mn0og/paella_dataset.tar.gz.part6?dl=0) (size ~12GB), and join them in one file as follow:
```
$ cat paella_dataset.tar.gz.part* > paella_dataset.tar.gz 
```
Then decompress it in the main folder `paella`:

```
$ paella_dataset.tar.gz | tar -zxf
```

The data are extracted to the `dataset` folder, which includes three subfolders, namely `dataset1`, `dataset2`, `dataset3`, each containing the respective dataset. Each dataset includes performance counters and Power Spectral Density measurements carried out with DiG [2]. For details on the performance metrics used in each dataset, please refer to the ReadMe files inside the respective sub-folders.

### Running the code

We tested the dataset we three different Machine Learning (ML) approaches, namely oc-SVM, IF and AE. You can find their implementions in the following files:

```
|-- 1_oc-SVM
|   `-- run_ocsvm.py
|-- 2_IF
|   `-- run_IF.py
|-- 3_AE
|   `-- run_AE.py 
```

On top of each file it is possible to set the parameters to use for the analysis. In particular, for oc-SVM and IF:

* *PATH* contains the dataset path (i.e., dataset1, dataset2 and dataset3).  
* *PATH_SCALER_STORE* and *PATH_SCALER_LOAD* contain the path where to store/load the scaler models.  
* *PATH_PCA_STORE* and *PATH_PCA_LOAD* contain the path where to store/load the PCA models.  
* *PATH_CLF_STORE* and *PATH_CLF_LOAD* contain the path where to store/load the ML models.  
* *METRIC* allows to set which metric to use for the analysis, namely set to 0=only_PSD_Welch, 1=only_examon (i.e., performance counters), 2=both  
* *TRAINING* allows to switch between training and inference phase, namely 1=Training, 0=Inference  
* *SCALER* is used to set the scaler, namely 0=None, 1=MinMax Scaler; 2=StandardScaler  
* *PCA* allows to enable or disable the Principal Component Analysis (PCA) to pre-process the data in input to ML models, namely namely 1=run_PCA, 0=do_not_run_PCA  
* *OUTLIER_TH* allows to set the threshold for percentage of outliers in the acquisition files. In other words, if the percentage of outliers in the acquisition is greater than this threshold, we label the acquisition as malware  
* *N_MALW_TO_TEST* allows to set the number of malware to use for test (min=1; max depends from the dataset used, namely max_dataset1=95, max_dataset2=6, max_dataset3=6)  
* *N_BENCH_TO_TEST* allows to set the number of benchmarks to use for test  (min=1, max=7, all benchmarks)  
* *N_TRAIN_FILES*, *N_VAL_FILES*, and *N_TST_FILES* correspond to the number of healthy benchmark acquisitions to use for training, validation and test, respectively. Notice that we used in our test 60% of samples for training, 20% for validation and 20% for test (namely N_TRAIN_FILES=18, N_VAL_FILES=6, and N_TST_FILES=6)  


Along with the previous parameters, the AE includes also:

* *PLOT_LOSS* allows to plot "Training vs. Validation loss", namely 0=do_not_plot_loss, 1=plot_loss  
* *ERROR* allows to set the error estimation method to use, namely 0=Root Mean Square Error (RMSE), 1=Mean Square Error (MSE), 2=Mean Absolute Error (MAE)  
* *RECONSTR_ERR_TH* is the threshold used for the reconstruction error, used to understand if the monitored metrics (i.e., performance counters metrics or PSDs) are outliers (e.g., if the RECONSTR_ERR_TH of a PSD is greater than 0.91, we label that PSD as an outlier)
* *EPOCH* is the epoch parameter to use for the AutoEncoder
* *BATCH* is the batch size to use for the AutoEncoder
* *OPTIMIZER* is the optimizer to use for the AutoEncoder (e.g., *adagrad*)
* *LOSS* is the loss function to use for AutoEncoder

Finally, to carry out the analysis, just run the python code, e.g.:

```
$ python run_AE.py
``` 

For questions, please send an email to a.libri@iis.ee.ethz.ch ([Antonio Libri, ETH Zurich](https://ee.ethz.ch/the-department/people-a-z/person-detail.MjIxODQ2.TGlzdC8zMjc5LC0xNjUwNTg5ODIw.html)).

### License and Attribution
Please refer to the LICENSE file for the licensing of our code.

