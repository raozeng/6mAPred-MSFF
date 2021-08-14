import matplotlib as mpl
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling1D
mpl.use('Agg')
from keras.models import load_model
from keras.models import Model
from keras.layers import Attention, Dense, Dropout, Conv1D, Conv2D,DepthwiseConv2D, Input, Reshape,MaxPooling1D,Flatten,LeakyReLU, ReLU, Embedding, Add, Multiply, LSTM, Bidirectional,Concatenate, GlobalAveragePooling1D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.activations import sigmoid
import random
import pandas as pd 
import numpy as np
from keras import regularizers
from keras.metrics import binary_accuracy
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os, sys, copy, getopt, re, argparse
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from keras import losses
import pickle
import csv

def dataProcessing(path,label, calculate,f_type):
    X = []
    y = []
    if f_type == 'csv':
        f = open(path)
        f_csv = csv.reader(f)
        #headers = next(f_csv)
        for row in f_csv:
            X.append(calculate(row))
            y.append(label)
    else:
        f = open(path)
        data = f.readlines()
        for line in data:
            if len(line) >= 41:
                line = list(line.strip('\n'));
                if 'N' not in line:
                    X.append(calculate(line))
                    y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    return X, y; #(n, 41), (n,1)

def calculate_K1(sequence):
    X = []
    dictNum = {'A' : 1, 'T' : 2, 'C' : 3, 'G' : 4, 'N':0}
    for s in sequence:
        X.append(dictNum[s])

    X = np.array(X)
    return X

def calculate_K1_NAC(sequence):
    X = []
    dictNum = {'A' : 1, 'T' : 2, 'C' : 3, 'G' : 4}
    dictNAC = {'A':0, 'T':1, 'C':0, 'G':0}
    for s in sequence:
        X.append(dictNum[s])
    for s in sequence:
        dictNAC[s] = dictNAC[s] + 1
    
    for k in dictNAC.keys():
        dictNAC[k] = dictNAC[k] * 1.0/41
        X.append(dictNAC[k])
    X = np.array(X)
    return X

def calculate_K2(sequence):
    X = []
    dictNum = { 'AA':1, 'AC':2, 'AG':3, 'AT':4, 'CA':5, 'CC':6, 'CG':7, 'CT':8, 'GA':9, 
            'GC':10, 'GG':11, 'GT':12, 'TA':13, 'TC':14, 'TG':15, 'TT':16,
            'NN':0,'AN':17,'CN':18,'GN':19,'TN':20,'NA':21,'NC':22,'NG':23,'NT':24}

    for index in range(len(sequence)-1):
        X.append(dictNum["".join(sequence[index:index+2])])

    X = np.array(X)
    return X

def calculate_K3(sequence):
    X = []
    dictNum = { 'AAA':1, 'AAC':2, 'AAG':3, 'AAT':4, 'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8, 'AGA':9, 
            'AGC':10, 'AGG':11, 'AGT':12, 'ATA':13, 'ATC':14, 'ATG':15, 'ATT':16,
            'CAA':17, 'CAC':18, 'CAG':19, 'CAT':20, 'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24, 'CGA':25, 
            'CGC':26, 'CGG':27, 'CGT':28, 'CTA':29, 'CTC':30, 'CTG':31, 'CTT':32,
            'GAA':33, 'GAC':34, 'GAG':35, 'GAT':36, 'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40, 'GGA':41, 
            'GGC':42, 'GGG':43, 'GGT':44, 'GTA':45, 'GTC':46, 'GTG':47, 'GTT':48,
            'TAA':49, 'TAC':50, 'TAG':51, 'TAT':52, 'TCA':53, 'TCC':54, 'TCG':55, 'TCT':56, 'TGA':57, 
            'TGC':58, 'TGG':59, 'TGT':60, 'TTA':61, 'TTC':62, 'TTG':63, 'TTT':64}

    for index in range(len(sequence)-2):
        X.append(dictNum["".join(sequence[index:index+3])])

    X = np.array(X)
    return X

def calculate_K2_DNC(sequence):
    X = []
    dictNum = { 'AA':1, 'AC':2, 'AG':3, 'AT':4, 'CA':5, 'CC':6, 'CG':7, 'CT':8, 'GA':9, 
            'GC':10, 'GG':11, 'GT':12, 'TA':13, 'TC':14, 'TG':15, 'TT':16}
    dictDNC = { 'AA':0, 'AC':0, 'AG':0, 'AT':0, 'CA':0, 'CC':0, 'CG':0, 'CT':0, 'GA':0, 
            'GC':0, 'GG':0, 'GT':0, 'TA':0, 'TC':0, 'TG':0, 'TT':0}

    for index in range(len(sequence)-1):
        X.append(dictNum["".join(sequence[index:index+2])])

    for index in range(len(sequence)-1):
        dictDNC["".join(sequence[index:index+2])] = dictDNC["".join(sequence[index:index+2])] + 1
    
    for k in dictDNC.keys():
        dictDNC[k] = dictDNC[k] * 1.0/40
        X.append(dictDNC[k])
    X = np.array(X)
    return X

def calculate_K1_NAC_K2_DNC(sequence):
    X = []
    dictNum = {'A' : 1, 'T' : 2, 'C' : 3, 'G' : 4}
    dictNAC = {'A':0, 'T':0, 'C':0, 'G':0}
    for s in sequence:
        X.append(dictNum[s])
    for s in sequence:
        dictNAC[s] = dictNAC[s] + 1
    
    for k in dictNAC.keys():
        dictNAC[k] = dictNAC[k] * 1.0/41
        X.append(dictNAC[k])

    dictNum2 = { 'AA':1, 'AC':2, 'AG':3, 'AT':4, 'CA':5, 'CC':6, 'CG':7, 'CT':8, 'GA':9, 
            'GC':10, 'GG':11, 'GT':12, 'TA':13, 'TC':14, 'TG':15, 'TT':16}
            #,'AN':17,'TN':18,'CN':19,'GN':20, 'NN':21, 'NA':22, 'NT':23,'NC':24,'NG':25}
    dictDNC = { 'AA':0, 'AC':0, 'AG':0, 'AT':0, 'CA':0, 'CC':0, 'CG':0, 'CT':0, 'GA':0, 
            'GC':0, 'GG':0, 'GT':0, 'TA':0, 'TC':0, 'TG':0, 'TT':0}
            #,'AN':0,'TN':0,'CN':0,'GN':0, 'NN':0, 'NA':0, 'NT':0,'NC':0,'NG':0}

    for index in range(len(sequence)-1):
        X.append(dictNum2["".join(sequence[index:index+2])])

    for index in range(len(sequence)-1):
        dictDNC["".join(sequence[index:index+2])] = dictDNC["".join(sequence[index:index+2])] + 1
    
    for k in dictDNC.keys():
        dictDNC[k] = dictDNC[k] * 1.0/40
        X.append(dictDNC[k])

    return X

def calculate_K1_K2(sequence):
    X = []
    dictNum = {'A' : 1, 'T' : 2, 'C' : 3, 'G' : 4}
    for s in sequence:
        X.append(dictNum[s])
   
    dictNum2 = { 'AA':1, 'AC':2, 'AG':3, 'AT':4, 'CA':5, 'CC':6, 'CG':7, 'CT':8, 'GA':9, 
            'GC':10, 'GG':11, 'GT':12, 'TA':13, 'TC':14, 'TG':15, 'TT':16}
            #,'AN':17,'TN':18,'CN':19,'GN':20, 'NN':21, 'NA':22, 'NT':23,'NC':24,'NG':25}
  
    for index in range(len(sequence)-1):
        X.append(dictNum2["".join(sequence[index:index+2])])

    return X

def calculate_K1_K3(sequence):
    X = []
    dictNum = {'A' : 1, 'T' : 2, 'C' : 3, 'G' : 4}
    for s in sequence:
        X.append(dictNum[s])
   
    dictNum3 = { 'AAA':1, 'AAC':2, 'AAG':3, 'AAT':4, 'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8, 'AGA':9, 
            'AGC':10, 'AGG':11, 'AGT':12, 'ATA':13, 'ATC':14, 'ATG':15, 'ATT':16,
            'CAA':17, 'CAC':18, 'CAG':19, 'CAT':20, 'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24, 'CGA':25, 
            'CGC':26, 'CGG':27, 'CGT':28, 'CTA':29, 'CTC':30, 'CTG':31, 'CTT':32,
            'GAA':33, 'GAC':34, 'GAG':35, 'GAT':36, 'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40, 'GGA':41, 
            'GGC':42, 'GGG':43, 'GGT':44, 'GTA':45, 'GTC':46, 'GTG':47, 'GTT':48,
            'TAA':49, 'TAC':50, 'TAG':51, 'TAT':52, 'TCA':53, 'TCC':54, 'TCG':55, 'TCT':56, 'TGA':57, 
            'TGC':58, 'TGG':59, 'TGT':60, 'TTA':61, 'TTC':62, 'TTG':63, 'TTT':64}
  
    for index in range(len(sequence)-2):
        X.append(dictNum3["".join(sequence[index:index+3])])

    return X

def calculate_K2_K3(sequence):
    X = []
    dictNum2 = { 'AA':1, 'AC':2, 'AG':3, 'AT':4, 'CA':5, 'CC':6, 'CG':7, 'CT':8, 'GA':9, 
            'GC':10, 'GG':11, 'GT':12, 'TA':13, 'TC':14, 'TG':15, 'TT':16}
            #,'AN':17,'TN':18,'CN':19,'GN':20, 'NN':21, 'NA':22, 'NT':23,'NC':24,'NG':25}
  
    for index in range(len(sequence)-1):
        X.append(dictNum2["".join(sequence[index:index+2])])
   
    dictNum3 = { 'AAA':1, 'AAC':2, 'AAG':3, 'AAT':4, 'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8, 'AGA':9
        , 'AGC':10, 'AGG':11, 'AGT':12, 'ATA':13, 'ATC':14, 'ATG':15, 'ATT':16
        ,'CAA':17, 'CAC':18, 'CAG':19, 'CAT':20, 'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24, 'CGA':25
        , 'CGC':26, 'CGG':27, 'CGT':28, 'CTA':29, 'CTC':30, 'CTG':31, 'CTT':32
        ,'GAA':33, 'GAC':34, 'GAG':35, 'GAT':36, 'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40, 'GGA':41
        , 'GGC':42, 'GGG':43, 'GGT':44, 'GTA':45, 'GTC':46, 'GTG':47, 'GTT':48
        ,'TAA':49, 'TAC':50, 'TAG':51, 'TAT':52, 'TCA':53, 'TCC':54, 'TCG':55, 'TCT':56, 'TGA':57
        , 'TGC':58, 'TGG':59, 'TGT':60, 'TTA':61, 'TTC':62, 'TTG':63, 'TTT':64}
  
    for index in range(len(sequence)-2):
        X.append(dictNum3["".join(sequence[index:index+3])])

    return X

def calculate_csv(sequence):
    X = []
    for item in sequence[1:]:
        X.append(float(item))
    X = np.array(X)
    return X

def calculate_NCP(sequence):
    X = []
    dictNum = {'A' : [1,1,1], 'T' : [0, 0, 1], 'C' : [0, 1, 0], 'G' : [1, 0, 0]}

    for s in sequence:
        X.append(dictNum[s])

    X = np.array(X)
    return X


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    
    return out

def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;

def shuffleDataX(X):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    return X;

def prepareData(pf, nf,calculate, f_type=False):
    Positive_X, Positive_y = dataProcessing(pf,1, calculate,f_type)
    Negitive_X, Negitive_y = dataProcessing(nf,0, calculate,f_type)

    return Positive_X, Positive_y, Negitive_X, Negitive_y

def ms_cam(x,shape):
    x_features_global = GlobalAveragePooling1D()(x) #(1,32)
    x_features_global = Reshape((1,x_features_global.shape[1]))(x_features_global)
    x_features_global = Conv1D(filters = shape[2]*6, kernel_size = 1,activation = 'elu',input_shape = x_features_global.shape)(x_features_global)
    x_features_global = BatchNormalization(epsilon=1.001e-5)(x_features_global)
    x_features_global = Conv1D(filters = shape[2], kernel_size = 1,activation = 'linear',input_shape =x_features_global.shape)(x_features_global)
    
    x_features_local = Conv1D(filters = shape[2]/6, kernel_size = 1,activation = 'elu',input_shape = shape)(x)
    x_features_local = Conv1D(filters = shape[2], kernel_size = 1,activation = 'linear',input_shape = x_features_local.shape)(x_features_local)
    x_features_local = BatchNormalization(epsilon=1.001e-5)(x_features_local)
    x_add = Add()([x_features_global,x_features_local])
    x_weight = sigmoid(x_add)
    return x_weight

def aff(x, y):
    x_y_add = Add()([x,y])
    weight = ms_cam(x_y_add, x_y_add.shape)
    x = Multiply()([x, 1+weight])
    y = Multiply()([y, weight])
    features = Add()([x, y])
    return features

def ConvN(n,x):
    k1_x = x
    for i in range(n):
        k1_x = Conv1D(filters = 256, kernel_size = 11,activation = 'elu',input_shape = x.shape)(k1_x)
        k1_x = ZeroPadding1D(padding=5)(k1_x)
        k1_x = BatchNormalization()(k1_x)
        k1_x = Dropout(0.5)(k1_x)

    return k1_x

def get6mAPred_MSFF():
    input_shape1 = (41)
    input_shape2 = (4)    
    input_shape3 = (40)
    input_shape4 = (16)

    inputs_k1 = Input(shape=input_shape1)
    inputs_k2 = Input(shape=input_shape2)
    inputs_k3 = Input(shape=input_shape3)
    inputs_k4 = Input(shape=input_shape4)

    embedding_layer1 = Embedding(input_dim=42, output_dim=64)
    embedding_layer2 = Embedding(input_dim=5, output_dim=64)
    embedding_layer3 = Embedding(input_dim=42, output_dim=64)
    embedding_layer4 = Embedding(input_dim=18, output_dim=64)
    lstm_layer = Bidirectional(LSTM(32,  return_sequences=True))
   

    k1_features = embedding_layer1(inputs_k1)
    k2_features = embedding_layer2(inputs_k2)
    k3_features = embedding_layer3(inputs_k3)
    k4_features = embedding_layer4(inputs_k4)

    k1 = Concatenate(axis=1)([k1_features,k2_features])
    k3 = Concatenate(axis=1)([k3_features,k4_features])

    k1_x = Conv1D(filters = 256, kernel_size = 1,activation = None,input_shape = k1.shape)(k1)
    k1_x = ConvN(6,k1_x)
    k1_x = Conv1D(filters = 64, kernel_size = 1,activation = None,input_shape = k1_x.shape)(k1_x)
    k1_x = Multiply()([k1_x,ms_cam(k1_x,k1_x.shape)])
    k1_x = lstm_layer(k1_x)

    k3_x = Conv1D(filters = 256, kernel_size = 1,activation = None,input_shape = k3.shape)(k3)
    k3_x = ConvN(6,k3_x)
    k3_x = Conv1D(filters = 64, kernel_size = 12,activation = None,input_shape = k3_x.shape)(k3_x)
    k3_x = Multiply()([k3_x,ms_cam(k3_x,k3_x.shape)])
    k3_x = lstm_layer(k3_x)

    features = aff(k1_x, k3_x)
    x = Flatten()(features)
    x = Dense(32, activation = 'elu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = [inputs_k1, inputs_k2,inputs_k3,inputs_k4], outputs = outputs)
    model.compile(loss='binary_crossentropy', optimizer= Adam(), metrics=[binary_accuracy])

    model.summary()
    return model

def getmodel_Kn(shape1,in_dim,em_dim):
    inputs_k1 = Input(shape=shape1)
    embedding_layer = Embedding(input_dim=in_dim, output_dim=em_dim)
    lstm_layer = Bidirectional(LSTM(32,  return_sequences=True))

    k1_features = embedding_layer(inputs_k1)
    #经过ms_cam 之后再 lstm
    k1_x = Conv1D(filters = 256, kernel_size = 1,activation = None,input_shape = k1_features.shape)(k1_features)
    k1_x = ConvN(6,k1_x)
    k1_x = Conv1D(filters = 64, kernel_size = 1,activation = None,input_shape = k1_x.shape)(k1_x)
    k1_x = Multiply()([k1_x,ms_cam(k1_x,k1_x.shape)])
    k1_x = lstm_layer(k1_x)
    x = Flatten()(k1_x)
    x = Dense(32, activation = 'elu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = inputs_k1, outputs = outputs)
    model.compile(loss='binary_crossentropy', optimizer= Adam(), metrics=[binary_accuracy])
    model.summary()
    return model

def getmodel_Dn(shape1,shape2,in_dim,em_dim, in_dim2, em_dim2):
    inputs_k1 = Input(shape=shape1)
    inputs_k2 = Input(shape=shape2)

    embedding_layer1 = Embedding(input_dim=in_dim, output_dim=em_dim)
    embedding_layer2 = Embedding(input_dim=in_dim2, output_dim=em_dim2)
    lstm_layer = Bidirectional(LSTM(32,  return_sequences=True))
   
    k1_features = embedding_layer1(inputs_k1)
    k2_features = embedding_layer2(inputs_k2)

    #经过ms_cam 之后再 lstm
    k1_x = Conv1D(filters = 256, kernel_size = 1,activation = None,input_shape = k1_features.shape)(k1_features)
    k1_x = ConvN(6,k1_x)
    k1_x = Conv1D(filters = 64, kernel_size = 2,activation = 'elu',input_shape = k1_x.shape)(k1_x)
    k1_x = Conv1D(filters = 64, kernel_size = 1,activation = None,input_shape = k1_x.shape)(k1_x)
    k1_x = Multiply()([k1_x,ms_cam(k1_x,k1_x.shape)])
    k1_x = lstm_layer(k1_x)

    k2_x = Conv1D(filters = 256, kernel_size = 1,activation = None,input_shape = k2_features.shape)(k2_features)
    k2_x = ConvN(6,k2_x)
    k2_x = Conv1D(filters = 64, kernel_size = 1,activation = None,input_shape = k2_x.shape)(k2_x)
    k2_x = Multiply()([k2_x,ms_cam(k2_x,k2_x.shape)])
    k2_x = lstm_layer(k2_x)
    
    features = aff(k1_x, k2_x)
    x = Flatten()(features)
    x = Dense(32, activation = 'elu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = [inputs_k1, inputs_k2], outputs = outputs)
    model.compile(loss='binary_crossentropy', optimizer= Adam(), metrics=[binary_accuracy])
    model.summary()
    return model

def getmodel_NE(shape1):
    inputs_k1 = Input(shape=shape1)
    lstm_layer = Bidirectional(LSTM(32,  return_sequences=True))
    #经过ms_cam 之后再 lstm
    k1_x = tf.expand_dims(inputs_k1,axis=-1)
    k1_x = Conv1D(filters = 256, kernel_size = 1,activation = None,input_shape = k1_x.shape)(k1_x)
    k1_x = ConvN(6,k1_x)
    k1_x = Conv1D(filters = 64, kernel_size = 1,activation = None,input_shape = k1_x.shape)(k1_x)
    k1_x = Multiply()([k1_x,ms_cam(k1_x,k1_x.shape)])
    k1_x = lstm_layer(k1_x)
    x = Flatten()(k1_x)
    x = Dense(32, activation = 'elu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = inputs_k1, outputs = outputs)
    model.compile(loss='binary_crossentropy', optimizer= Adam(), metrics=[binary_accuracy])
    model.summary()
    return model

# train 6mAPred_MSFF by the proposed encoding schema
def train_test(pf,nf,calculate,output_dir, folds):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate)
    # Positive_X = shuffleDataX(Positive_X)
    # Negitive_X = shuffleDataX(Negitive_X)
    Positive_X_Slices = chunkIt(Positive_X, folds);
    Positive_y_Slices = chunkIt(Positive_y, folds);

    Negative_X_Slices = chunkIt(Negitive_X, folds);
    Negative_y_Slices = chunkIt(Negitive_y, folds);

    trainning_result = []
    validation_result = []
    testing_result = []

    for test_index in range(folds):
        test_X = np.concatenate((Positive_X_Slices[test_index],Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index],Negative_y_Slices[test_index]))

        validation_index = (test_index+1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index],Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index],Negative_y_Slices[validation_index]))

        start = 0

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start],Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start],Negative_y_Slices[start]))

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i],Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i],Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))
               
        test_X, test_y = shuffleData(test_X,test_y)
        valid_X,valid_y = shuffleData(valid_X,valid_y)
        train_X,train_y = shuffleData(train_X,train_y)

        test_X_k1 = np.array([item[0:41] for item in test_X])
        test_X_k2 = np.array([item[41:45] for item in test_X])
        test_X_k3 = np.array([item[45:85] for item in test_X])
        test_X_k4 = np.array([item[85:] for item in test_X])

        valid_X_k1 = np.array([item[0:41] for item in valid_X])
        valid_X_k2 = np.array([item[41:45] for item in valid_X])
        valid_X_k3 = np.array([item[45:85] for item in valid_X])
        valid_X_k4 = np.array([item[85:] for item in valid_X])
        
        train_X_k1 = np.array([item[0:41] for item in train_X])
        train_X_k2 = np.array([item[41:45] for item in train_X])
        train_X_k3 = np.array([item[45:85] for item in train_X])
        train_X_k4 = np.array([item[85:] for item in train_X])


        model = get6mAPred_MSFF()
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience= 30)
        model_check = ModelCheckpoint(filepath = output_dir + "/model" + str(test_index+1) +".h5", monitor = 'val_binary_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=20)

        history = model.fit([train_X_k1,train_X_k2,train_X_k3,train_X_k4], train_y, batch_size = 80, epochs = 50
                    , validation_data = ([valid_X_k1,valid_X_k2,valid_X_k3,valid_X_k4], valid_y),callbacks = [model_check, early_stopping,reduct_L_rate]);    

#  envalute 6mAPred_MSFF by the proposed encoding schema
def test(model_dir,pf,nf,calculate,output_dir, sample_size=False):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate)

    if sample_size != False:
        Positive_X = random.sample(list(Positive_X), sample_size)
        Negitive_X = random.sample(list(Negitive_X), sample_size)

        Positive_y = random.sample(list(Positive_y), sample_size)
        Negitive_y = random.sample(list(Negitive_y), sample_size)

    test_X = np.concatenate((Positive_X,Negitive_X))
    test_y = np.concatenate((Positive_y,Negitive_y))

    test_X, test_y = shuffleData(test_X,test_y);

    test_X_k1 = np.array([item[0:41] for item in test_X])
    test_X_k2 = np.array([item[41:45] for item in test_X])
    test_X_k3 = np.array([item[45:85] for item in test_X])
    test_X_k4 = np.array([item[85:] for item in test_X])


    model = load_model(model_dir)
    score = model.evaluate([test_X_k1,test_X_k2,test_X_k3,test_X_k4],test_y)
    pred_y = model.predict([test_X_k1,test_X_k2,test_X_k3,test_X_k4])

    pf = open(output_dir+'/pos.txt','+w')
    nf = open(output_dir+'/neg.txt','+w')
    for index in range(len(test_y)):
        if test_y[index] == 0:
            str_neg = "nsample%s\t%s\n" % (index, pred_y[index][0])
            nf.write(str_neg)
        if test_y[index] == 1:
            str_pos = "psample%s\t%s\n" % (index, pred_y[index][0])
            pf.write(str_pos)

# train 6mAPred_MSFF by 1-gram, 2-grams and 3-grams
def train_test_Kn(pf,nf,calculate,output_dir, folds, in_dim,em_dim):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate)

    Positive_X_Slices = chunkIt(Positive_X, folds);
    Positive_y_Slices = chunkIt(Positive_y, folds);

    Negative_X_Slices = chunkIt(Negitive_X, folds);
    Negative_y_Slices = chunkIt(Negitive_y, folds);

    trainning_result = []
    validation_result = []
    testing_result = []

    for test_index in range(folds):
        test_X = np.concatenate((Positive_X_Slices[test_index],Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index],Negative_y_Slices[test_index]))

        validation_index = (test_index+1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index],Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index],Negative_y_Slices[validation_index]))

        start = 0

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start],Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start],Negative_y_Slices[start]))

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i],Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i],Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        test_X, test_y = shuffleData(test_X,test_y)
        valid_X,valid_y = shuffleData(valid_X,valid_y)
        train_X,train_y = shuffleData(train_X,train_y)

        model = getmodel_Kn(train_X.shape[1:],in_dim,em_dim)
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience= 30)
        model_check = ModelCheckpoint(filepath = output_dir + "/model" + str(test_index+1) +".h5", monitor = 'val_binary_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=20)

        history = model.fit(train_X, train_y, batch_size = 200, epochs = 50, validation_data = (valid_X, valid_y),callbacks = [model_check, early_stopping,reduct_L_rate]);    

# envalute 6mAPred_MSFF by 1-gram, 2-grams and 3-grams
def test_Kn(model_dir,pf,nf,calculate,output_dir, sample_size=False):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate)

    if sample_size != False:
        Positive_X = random.sample(list(Positive_X), sample_size)
        Negitive_X = random.sample(list(Negitive_X), sample_size)

        Positive_y = random.sample(list(Positive_y), sample_size)
        Negitive_y = random.sample(list(Negitive_y), sample_size)

    test_X = np.concatenate((Positive_X,Negitive_X))
    test_y = np.concatenate((Positive_y,Negitive_y))


    model = load_model(model_dir)
    score = model.evaluate(test_X,test_y)
    pred_y = model.predict(test_X)

    pf = open(output_dir+'/pos.txt','+w')
    nf = open(output_dir+'/neg.txt','+w')
    for index in range(len(test_y)):
        if test_y[index] == 0:
            str_neg = "nsample%s\t%s\n" % (index, pred_y[index][0])
            nf.write(str_neg)
        if test_y[index] == 1:
            str_pos = "psample%s\t%s\n" % (index, pred_y[index][0])
            pf.write(str_pos)

# train 6mAPred_MSFF by 1-gram and 2-grams, 1-gram and 3-grams, 2-grams and 3-grams
def train_test_Dn(pf,nf,calculate,output_dir, folds,split_pos,in_dim,em_dim, in_dim2, em_dim2):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate)

    Positive_X_Slices = chunkIt(Positive_X, folds);
    Positive_y_Slices = chunkIt(Positive_y, folds);

    Negative_X_Slices = chunkIt(Negitive_X, folds);
    Negative_y_Slices = chunkIt(Negitive_y, folds);

    for test_index in range(folds):
        test_X = np.concatenate((Positive_X_Slices[test_index],Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index],Negative_y_Slices[test_index]))

        validation_index = (test_index+1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index],Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index],Negative_y_Slices[validation_index]))

        start = 0

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start],Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start],Negative_y_Slices[start]))

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i],Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i],Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        test_X, test_y = shuffleData(test_X,test_y)
        valid_X,valid_y = shuffleData(valid_X,valid_y)
        train_X,train_y = shuffleData(train_X,train_y)

        test_X_k1 = np.array([item[0:split_pos] for item in test_X])
        test_X_k2 = np.array([item[split_pos:] for item in test_X])

        valid_X_k1 = np.array([item[0:split_pos] for item in valid_X])
        valid_X_k2 = np.array([item[split_pos:] for item in valid_X])

        train_X_k1 = np.array([item[0:split_pos] for item in train_X])
        train_X_k2 = np.array([item[split_pos:] for item in train_X])

        model = getmodel_Dn(train_X_k1.shape[1:],train_X_k2.shape[1:],in_dim,em_dim, in_dim2, em_dim2)
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience= 30)
        model_check = ModelCheckpoint(filepath = output_dir + "/model" + str(test_index+1) +".h5", monitor = 'val_binary_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=20)

        history = model.fit([train_X_k1,train_X_k2], train_y, batch_size = 80, epochs = 60, validation_data = ([valid_X_k1,valid_X_k2], valid_y),callbacks = [model_check, early_stopping,reduct_L_rate]);    

# test 6mAPred_MSFF by 1-gram and 2-grams, 1-gram and 3-grams, 2-grams and 3-grams
def test_Dn(model_dir,pf,nf,calculate,output_dir,split_pos, sample_size=False):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate)

    if sample_size != False:
        Positive_X = random.sample(list(Positive_X), sample_size)
        Negitive_X = random.sample(list(Negitive_X), sample_size)

        Positive_y = random.sample(list(Positive_y), sample_size)
        Negitive_y = random.sample(list(Negitive_y), sample_size)

    test_X = np.concatenate((Positive_X,Negitive_X))
    test_y = np.concatenate((Positive_y,Negitive_y))

    test_X_k1 = np.array([item[0:split_pos] for item in test_X])
    test_X_k2 = np.array([item[split_pos:] for item in test_X])

    model = load_model(model_dir)
    score = model.evaluate([test_X_k1,test_X_k2],test_y)
    pred_y = model.predict([test_X_k1,test_X_k2])

    pf = open(output_dir+'/pos.txt','+w')
    nf = open(output_dir+'/neg.txt','+w')
    for index in range(len(test_y)):
        if test_y[index] == 0:
            str_neg = "nsample%s\t%s\n" % (index, pred_y[index][0])
            nf.write(str_neg)
        if test_y[index] == 1:
            str_pos = "psample%s\t%s\n" % (index, pred_y[index][0])
            pf.write(str_pos)

# train 6mAPred_MSFF by different feature descriptors
def train_test_NE(pf,nf,calculate,output_dir, folds):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate,'csv')

    Positive_X_Slices = chunkIt(Positive_X, folds);
    Positive_y_Slices = chunkIt(Positive_y, folds);

    Negative_X_Slices = chunkIt(Negitive_X, folds);
    Negative_y_Slices = chunkIt(Negitive_y, folds);

    trainning_result = []
    validation_result = []
    testing_result = []

    for test_index in range(folds):
        test_X = np.concatenate((Positive_X_Slices[test_index],Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index],Negative_y_Slices[test_index]))

        validation_index = (test_index+1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index],Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index],Negative_y_Slices[validation_index]))

        start = 0

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start],Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start],Negative_y_Slices[start]))

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i],Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i],Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        test_X, test_y = shuffleData(test_X,test_y)
        valid_X,valid_y = shuffleData(valid_X,valid_y)
        train_X,train_y = shuffleData(train_X,train_y)
        
        X = np.concatenate((train_X,valid_X,test_X))
        y = np.concatenate((train_y,valid_y,test_y))

        model = getmodel_NE(train_X.shape[1:])
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience= 30)
        model_check = ModelCheckpoint(filepath = output_dir + "/model" + str(test_index+1) +".h5", monitor = 'val_binary_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=20)

        history = model.fit(train_X, train_y, batch_size = 80, epochs = 50, validation_data = (X, y),callbacks = [model_check, early_stopping,reduct_L_rate]);    

# envaluate 6mAPred_MSFF by different feature descriptors
def test_NE(model_dir,pf,nf,calculate,output_dir, sample_size=False):
    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(pf,nf,calculate,'csv')

    if sample_size != False:
        Positive_X = random.sample(list(Positive_X), sample_size)
        Negitive_X = random.sample(list(Negitive_X), sample_size)

        Positive_y = random.sample(list(Positive_y), sample_size)
        Negitive_y = random.sample(list(Negitive_y), sample_size)

    test_X = np.concatenate((Positive_X,Negitive_X))
    test_y = np.concatenate((Positive_y,Negitive_y))


    model = load_model(model_dir)
    score = model.evaluate(test_X,test_y)
    pred_y = model.predict(test_X)

    pf = open(output_dir+'/pos.txt','+w')
    nf = open(output_dir+'/neg.txt','+w')
    for index in range(len(test_y)):
        if test_y[index] == 0:
            str_neg = "nsample%s\t%s\n" % (index, pred_y[index][0])
            nf.write(str_neg)
        if test_y[index] == 1:
            str_pos = "psample%s\t%s\n" % (index, pred_y[index][0])
            pf.write(str_pos)

            
def main():
 
    # train 6mAPred_MSFF by the proposed encoding schema based on the 6mA-rice-Lv dataset
    train_test('6mADataSets/RiceLv/pos.txt','6mADataSets/RiceLv/neg.txt'
                , calculate_K1_NAC_K2_DNC, '6mAPred_MSFF_OUT/model_DIR', 5)
    # envalute 6mAPred_MSFF by the proposed encoding schema based on the 6mA-rice-Lv dataset
    test('6mAPred_MSFF_OUT/model_DIR','6mADataSets/RiceLv/pos.txt','6mADataSets/RiceLv/neg.txt'
                , calculate_K1_NAC_K2_DNC, '6mAPred_MSFF_OUT/RiceLv/output') 

    # train 6mAPred_MSFF by the 2-grams encoding schema based on the 6mA-rice-Lv dataset
    train_test_Kn('6mADataSets/RiceLv/pos.txt','6mADataSets/RiceLv/neg.txt'
               , calculate_K2, '6mAPred_MSFF_OUT/model_DIR', 5,40,64)
    # envalute 6mAPred_MSFF by the 2-grams encoding schema based on the 6mA-rice-Lv dataset
    test_Kn('6mAPred_MSFF_OUT/model_DIR','6mADataSets/RiceLv/pos.txt','6mADataSets/RiceLv/neg.txt'
                , calculate_K2, '6mAPred_MSFF_OUT/RiceLv/K2/output/')
    
    # train 6mAPred_MSFF by the 1-gram and 2-grams encoding schema based on the 6mA-rice-Lv dataset
    train_test_Dn('6mADataSets/RiceLv/pos.txt','6mADataSets/RiceLv/neg.txt'
               , calculate_K1_K2, '6mAPred_MSFF_OUT/model_DIR', 5,41,41,64,40,64)
    # envalute 6mAPred_MSFF by the 1-gram and 2-grams encoding schema based on the 6mA-rice-Lv dataset
    test_Dn('6mAPred_MSFF_OUT/model_DIR','6mADataSets/RiceLv/pos.txt','6mADataSets/RiceLv/neg.txt'
                , calculate_K1_K2, '6mAPred_MSFF_OUT/RiceLv/K1_K2/output/',41)

    # train 6mAPred_MSFF by the NCP feature based on the 6mA-rice-Lv dataset
    train_test_NE('6mAFeatureCode/RiceLv/NCP/pos_code.txt','6mAFeatureCode/RiceLv/NCP/neg_code.txt'
               , calculate_csv, '6mAPred_MSFF_OUT/model_DIR', 5)
    # envalute 6mAPred_MSFF by the NCP feature based on the 6mA-rice-Lv dataset
    test_NE('6mAPred_MSFF_OUT/model_DIR'
            ,'6mAFeatureCode/RiceLv/NCP/pos_code.txt','6mAFeatureCode/RiceLv/NCP/neg_code.txt'
            , calculate_csv, '6mAPred_MSFF_OUT/RiceLv/NCP/output/')


if __name__ == "__main__":
    main()
