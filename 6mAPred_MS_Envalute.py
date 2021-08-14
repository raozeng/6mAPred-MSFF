import matplotlib as mpl
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling1D
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
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc,precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import os, sys, copy, getopt, re, argparse
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from keras import losses
import pickle
import csv
from scipy import interp


def calculateScoreFile(pos_file, neg_file,split_char='\t'):
    pf = open(pos_file)
    nf = open(neg_file)

    p_data = pf.readlines()
    n_data = nf.readlines()
    pred_y = []
    y = []
    accuracy = 0
    correct = 0

    for pline in p_data:
        item = float(pline.split(split_char)[1])
        pred_y.append(item)
        y.append(1)
        if item >= 0.5:
            correct = correct +1

    for nline in n_data:
        item = float(nline.split(split_char)[1])
        pred_y.append(item)
        y.append(0)
        if item < 0.5:
            correct = correct + 1
    
    accuracy = correct *1.0/ (len(y))
    
    y = np.array(y)
    pred_y = np.array(pred_y)

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()

    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN+FP)
    MCC = matthews_corrcoef(y, tempLabel)

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)
    

    pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    precision_pr, recall, thresholds_pr = precision_recall_curve(y, pred_y)
    AP = average_precision_score(y, pred_y)

    print(y.shape)
    print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)

    lossValue = 123
    return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea
    , 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds
    ,'precision_pr':precision_pr,'recall':recall, 'thresholds_pr':thresholds_pr,'AP':AP, 'lossValue' : lossValue}


def calculateScoreML(train_file, test_file,split_char='\t'):
    trainf = open(train_file)
    testf = open(test_file)

    trian_data = trainf.readlines()
    test_data = testf.readlines()
    pred_y = []
    y = []
    accuracy = 0
    correct = 0

    for line in trian_data:
        if '#' not in line:
            item = float(line.split(split_char)[1])
            item2 = int(line.split(split_char)[0])
            pred_y.append(item)
            y.append(item2)
            if item >= 0.5 and item2 == 1:
                correct = correct +1

    for line in test_data:
        if '#' not in line:
            item = float(line.split(split_char)[1])
            item2 = int(line.split(split_char)[0])
            pred_y.append(item)
            y.append(item2)
            if item < 0.5 and item2 == 0:
                correct = correct + 1
        
    accuracy = correct *1.0/ (len(y))
    
    y = np.array(y)
    pred_y = np.array(pred_y)

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()

    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN+FP)
    MCC = matthews_corrcoef(y, tempLabel)

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)
    

    pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    precision_pr, recall, thresholds_pr = precision_recall_curve(y, pred_y)
    AP = average_precision_score(y, pred_y)

    print(y.shape)
    print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    lossValue = 123
    return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea
    , 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds
    ,'precision_pr':precision_pr,'recall':recall, 'thresholds_pr':thresholds_pr,'AP':AP, 'lossValue' : lossValue}


def draw_ROC(result, OutputDir, title):    
    for key in result.keys():
        val = result[key]
        tpr = val['tpr']
        fpr = val['fpr']
        roc_auc = auc(fpr, tpr)
        if key == '6mAPred_MSC':
            plt.plot(fpr, tpr, lw=2, alpha=1,label='%s (area = %0.3f)' % (key, roc_auc))
        else:
            plt.plot(fpr, tpr, lw=1.5, alpha=0.9,label='%s (area = %0.3f)' % (key, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title(title,fontsize=14)
    plt.legend(loc="lower right",fontsize=14)


    plt.savefig( OutputDir + '/' + title +'ROC.png')
    plt.close('all');

def draw_PR(result, OutputDir, title):
    for key in result.keys():        
        val = result[key]
        precision = val['precision_pr']
        recall = val['recall']
        ap = val['AP']
        recall = np.sort(recall)
        precision = np.sort(precision)[::-1]
        if key == 'SNNRice6mA-large':
            print(recall)
            print(precision)
        if key == '6mAPred_MSC':
            plt.plot(recall, precision, lw=2, alpha=1,label='%s (area = %0.3f)' % (key, ap))
        else:
            plt.plot(recall, precision, lw=1.5, alpha=0.9,label='%s (area = %0.3f)' % (key, ap))
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(title,fontsize=14)
    plt.legend(loc="lower left", fontsize=14)


    plt.savefig( OutputDir + '/' + title +'PR.png')
    plt.close('all');

def save_performance(result,output_file):
    f = open(output_file,'+w')
    content = ""
    for key in result.keys():
        val = result[key]
        content = "%s:  sn(%s),sp(%s),acc(%s),mcc(%s),auc(%s)\n" % (key,val['sn'],val['sp'],val['acc'],val['MCC'],val['AUC'])
        f.write(content)

def calculate_riceLv():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/6mAPred_MS/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/6mAPred_MS/neg_output.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/Deep6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/Deep6mA/neg_output.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/SNNRice6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/SNNRice6mA/neg_output.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/MM-6mAPred/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/MM-6mAPred/neg_output.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv','Rice_Lv')
    draw_PR(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv','Rice_Lv')
    save_performance(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceLv/performance.txt')

def calculate_riceChen():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/6mAPred_MS/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/6mAPred_MS/neg_output.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/Deep6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/Deep6mA/neg_output.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/SNNRice6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/SNNRice6mA/neg_output.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/MM-6mAPred/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/MM-6mAPred/neg_output.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen','Rice_Chen')
    draw_PR(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen','Rice_Chen')
    save_performance(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/RiceChen/performance.txt')

def calculate_A_thaliana():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/6mAPred_MS/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/6mAPred_MS/neg_output.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/Deep6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/Deep6mA/neg_output.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/SNNRice6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/SNNRice6mA/neg_output.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/MM-6mAPred/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/MM-6mAPred/neg_output.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana','A.thaliana')
    draw_PR(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana','A.thaliana')
    save_performance(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/A.thaliana/performance.txt')

def calculate_R_chinensis():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/6mAPred_MS/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/6mAPred_MS/neg_output.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/Deep6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/Deep6mA/neg_output.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/SNNRice6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/SNNRice6mA/neg_output.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/MM-6mAPred/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/MM-6mAPred/neg_output.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis','R.chinensis')
    draw_PR(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis','R.chinensis')
    save_performance(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/R.chinensis/performance.txt')


def calculate_F_vesca():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/6mAPred_MS/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/6mAPred_MS/neg_output.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/Deep6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/Deep6mA/neg_output.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/SNNRice6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/SNNRice6mA/neg_output.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/MM-6mAPred/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/MM-6mAPred/neg_output.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca','F.vesca')
    draw_PR(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca','F.vesca')
    save_performance(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/F.vesca/performance.txt')


def calculate_H_sapiens():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/6mAPred_MS/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/6mAPred_MS/neg_output.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/Deep6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/Deep6mA/neg_output.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/SNNRice6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/SNNRice6mA/neg_output.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/MM-6mAPred/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/MM-6mAPred/neg_output.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens','H.sapiens')
    draw_PR(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens','H.sapiens')
    save_performance(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/H.sapiens/performance.txt')

def calculate_D_melanogaster():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/6mAPred_MS/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/6mAPred_MS/neg_output.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/Deep6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/Deep6mA/neg_output.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/SNNRice6mA/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/SNNRice6mA/neg_output.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/MM-6mAPred/pos_output.txt','6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/MM-6mAPred/neg_output.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster','D.melanogaster')
    draw_PR(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster','D.melanogaster')
    save_performance(result,'6mAPred_MSFF_OUT/7_benchmark_5_fold/D.melanogaster/performance.txt')


def calculate_cross_A_thaliana():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/6mAPred_MS/neg.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/Deep6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/Deep6mA/neg.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/SNNRice6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/SNNRice6mA/neg.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/MM-6mAPred/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/MM-6mAPred/neg.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana','A.thaliana')
    draw_PR(result,'6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana','A.thaliana')
    save_performance(result,'6mAPred_MSFF_OUT/6_benchmark_cross/A.thaliana/performance.txt')

def calculate_cross_R_chinensis():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/6mAPred_MS/neg.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/Deep6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/Deep6mA/neg.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/SNNRice6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/SNNRice6mA/neg.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/MM-6mAPred/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/MM-6mAPred/neg.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis','R.chinensis')
    draw_PR(result,'6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis','R.chinensis')
    save_performance(result,'6mAPred_MSFF_OUT/6_benchmark_cross/R.chinensis/performance.txt')

def calculate_cross_F_vesca():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/6mAPred_MS/neg.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/Deep6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/Deep6mA/neg.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/SNNRice6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/SNNRice6mA/neg.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/MM-6mAPred/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/MM-6mAPred/neg.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca','F.vesca')
    draw_PR(result,'6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca','F.vesca')
    save_performance(result,'6mAPred_MSFF_OUT/6_benchmark_cross/F.vesca/performance.txt')

def calculate_cross_H_sapiens():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/6mAPred_MS/neg.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/Deep6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/Deep6mA/neg.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/SNNRice6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/SNNRice6mA/neg.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/MM-6mAPred/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/MM-6mAPred/neg.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens','H.sapiens')
    draw_PR(result,'6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens','H.sapiens')
    save_performance(result,'6mAPred_MSFF_OUT/6_benchmark_cross/H.sapiens/performance.txt')

def calculate_cross_D_melanogaster():
    msff = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/6mAPred_MS/neg.txt')
    deep6ma = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/Deep6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/Deep6mA/neg.txt')
    snn = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/SNNRice6mA/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/SNNRice6mA/neg.txt')
    mm = calculateScoreFile('6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/MM-6mAPred/pos.txt','6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/MM-6mAPred/neg.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma, 'SNNRice6mA-large':snn, 'MM-6mAPred':mm}
    draw_ROC(result,'6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster','D.melanogaster')
    draw_PR(result,'6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster','D.melanogaster')
    save_performance(result,'6mAPred_MSFF_OUT/6_benchmark_cross/D.melanogaster/performance.txt')

def calculate_n_grams():
    k1 = calculateScoreFile('6mAPred_MSFF_OUT/n-grams/K1/pos.txt','6mAPred_MSFF_OUT/n-grams/K1/neg.txt')
    k2 =  calculateScoreFile('6mAPred_MSFF_OUT/n-grams/K2/pos.txt','6mAPred_MSFF_OUT/n-grams/K2/neg.txt')
    k3 =  calculateScoreFile('6mAPred_MSFF_OUT/n-grams/K3/pos.txt','6mAPred_MSFF_OUT/n-grams/K3/neg.txt')
    k1k2 =  calculateScoreFile('6mAPred_MSFF_OUT/n-grams/K1K2/pos.txt','6mAPred_MSFF_OUT/n-grams/K1K2/neg.txt')
    k1k3 =  calculateScoreFile('6mAPred_MSFF_OUT/n-grams/K1K3/pos.txt','6mAPred_MSFF_OUT/n-grams/K1K3/neg.txt')
    k2k3 =  calculateScoreFile('6mAPred_MSFF_OUT/n-grams/K2K3/pos.txt','6mAPred_MSFF_OUT/n-grams/K2K3/neg.txt')
    result = {'1-gram':k1,'2-grams':k2, '3-grams':k3,'1-gram and 2grams':k1k2, '1-gram and 3-grams':k1k3, '2-grams and 3-grams':k2k3}
    draw_ROC(result,'6mAPred_MSFF_OUT/n-grams','n-grams')
    draw_PR(result,'6mAPred_MSFF_OUT/n-grams','n-grams')
    save_performance(result,'6mAPred_MSFF_OUT/n-grams/performance.txt')

def calculate_Kmer():
    msff  = calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/Kmer/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/Kmer/6mAPred_MS/neg.txt')
    deep6ma =  calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/Kmer/Deep6mA/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/Kmer/Deep6mA/neg.txt')
    rf = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/Kmer/RF/RF_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/Kmer/RF/RF_IND.txt')
    ann = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/Kmer/ANN/ANN_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/Kmer/ANN/ANN_IND.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma,'RF':rf, 'ANN': ann}
    draw_ROC(result,'6mAPred_MSFF_OUT/features/RiceLv/Kmer','Kmer')
    draw_PR(result,'6mAPred_MSFF_OUT/features/RiceLv/Kmer','Kmer')
    save_performance(result,'6mAPred_MSFF_OUT/features/RiceLv/Kmer/performance.txt')
    

def calculate_NCP():
    msff  = calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/NCP/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/NCP/6mAPred_MS/neg.txt')
    deep6ma =  calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/NCP/Deep6mA/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/NCP/Deep6mA/neg.txt')
    rf = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/NCP/RF/RF_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/NCP/RF/RF_IND.txt')
    ann = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/NCP/ANN/ANN_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/NCP/ANN/ANN_IND.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma,'RF':rf, 'ANN': ann}
    draw_ROC(result,'6mAPred_MSFF_OUT/features/RiceLv/NCP','NCP')
    draw_PR(result,'6mAPred_MSFF_OUT/features/RiceLv/NCP','NCP')
    save_performance(result,'6mAPred_MSFF_OUT/features/RiceLv/NCP/performance.txt')

def calculate_ENAC():
    msff  = calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/ENAC/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/ENAC/6mAPred_MS/neg.txt')
    deep6ma =  calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/ENAC/Deep6mA/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/ENAC/Deep6mA/neg.txt')
    rf = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/ENAC/RF/RF_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/ENAC/RF/RF_IND.txt')
    ann = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/ENAC/ANN/ANN_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/ENAC/ANN/ANN_IND.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma,'RF':rf, 'ANN': ann}
    draw_ROC(result,'6mAPred_MSFF_OUT/features/RiceLv/ENAC','ENAC')
    draw_PR(result,'6mAPred_MSFF_OUT/features/RiceLv/ENAC','ENAC')
    save_performance(result,'6mAPred_MSFF_OUT/features/RiceLv/ENAC/performance.txt')

def calculate_EIIP():
    msff  = calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/EIIP/6mAPred_MS/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/EIIP/6mAPred_MS/neg.txt')
    deep6ma =  calculateScoreFile('6mAPred_MSFF_OUT/features/RiceLv/EIIP/Deep6mA/pos.txt','6mAPred_MSFF_OUT/features/RiceLv/EIIP/Deep6mA/neg.txt')
    rf = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/EIIP/RF/RF_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/EIIP/RF/RF_IND.txt')
    ann = calculateScoreML('6mAPred_MSFF_OUT/features/RiceLv/EIIP/ANN/ANN_CV.txt','6mAPred_MSFF_OUT/features/RiceLv/EIIP/ANN/ANN_IND.txt')
    result = {'6mAPred_MSFF':msff,'Deep6mA':deep6ma,'RF':rf, 'ANN': ann}
    draw_ROC(result,'6mAPred_MSFF_OUT/features/RiceLv/EIIP','EIIP')
    draw_PR(result,'6mAPred_MSFF_OUT/features/RiceLv/EIIP','EIIP')
    save_performance(result,'6mAPred_MSFF_OUT/features/RiceLv/EIIP/performance.txt')


def main():
    calculate_riceLv()
    calculate_riceChen()
    calculate_A_thaliana()
    calculate_R_chinensis()
    calculate_F_vesca()
    calculate_H_sapiens()
    calculate_D_melanogaster()
    calculate_cross_A_thaliana()
    calculate_cross_R_chinensis()
    calculate_cross_F_vesca()
    calculate_cross_H_sapiens()
    calculate_cross_D_melanogaster()
    calculate_n_grams()
    calculate_Kmer()
    calculate_NCP()
    calculate_EIIP()
    calculate_ENAC()

if __name__ == "__main__":
    main()
