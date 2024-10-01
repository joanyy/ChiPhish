import tensorflow as tf 
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,Dropout,MaxPooling1D,Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def load_dataset(json_file):
    json_f=open(json_file).read() 
    json_info=json.loads(json_f)
    json_data=pd.json_normalize(json_info) # already normalization
    #print('json_data shape',json_data.shape)
    #print('json_Data columns',json_data.columns)
    return json_data
def get_fpr(y,y_pred):   
    cm = confusion_matrix(y, y_pred)
    tn=cm[0, 0]
    fp=cm[0, 1]
    fn=cm[1, 0]
    tp=cm[1, 1]
    '''
    print('tn',tn)
    print('tp',tp)
    print('fp',fp)
    print('fn',fn)
    '''
    #fpr=fp/(fp+tn)
    tpr=tp/(tp+fn)
    tnr=tn/(tn+fp)
    f1_score=2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))
    #tnr=format(tnr,'.4f')
    #tpr=format(tpr,'.2f')
    return tnr,tpr,f1_score
def load_dataset(json_file):
    json_f=open(json_file).read()
    json_info=json.loads(json_f)
    json_data=pd.json_normalize(json_info) # already normalization
    #print('json_data shape',json_data.shape)
    #print('json_Data columns',json_data.columns)
    return json_data
def get_fpr(y,y_pred):
    cm = confusion_matrix(y, y_pred)
    tn=cm[0, 0]
    fp=cm[0, 1]
    fn=cm[1, 0]
    tp=cm[1, 1]
    '''
    print('tn',tn)
    print('tp',tp)
    print('fp',fp)
    print('fn',fn)
    '''
    #fpr=fp/(fp+tn)
    tpr=tp/(tp+fn)
    tnr=tn/(tn+fp)
    f1_score=2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))
    return tnr,tpr,f1_score

def full_metri(model,train_x,train_y,test_x,test_y):
    start=time.time()
    model.fit(train_x,train_y)
    end=time.time()
    time1=(end-start)#second
    pred=model.predict(test_x)
    tnr,tpr,f1_score=get_fpr(test_y,pred)
    print('recall is %.2f,tnr is %.2f'%(tpr,tnr))
    return model,tnr,tpr,time1,f1_score


def cnn_metri(model,test_x,test_y):
    predict_y_html=model.predict(test_x)
    pre_class_html=np.argmax(predict_y_html,axis=1)
    confusion_mtx_html = confusion_matrix(test_y, pre_class_html)
    #print(confusion_mtx_com)
    cnn_tnr_html,cnn_recall_html,cnn_f1_html=get_fpr(test_y,pre_class_html)
    print('recall is %.2f, tnr is %.2f' %(cnn_recall_html,cnn_tnr_html))
    return cnn_tnr_html,cnn_recall_html,cnn_f1_html

def pre_metri(model,test_x,test_y):
    pred=model.predict(test_x)
    tnr,tpr,f1_score=get_fpr(test_y,pred)
    print('recall is %.2f, tnr is %.2f'%(tpr,tnr))
    return tnr,tpr,f1_score
#for chiphish_spacephishF

def create_model_cnn_full1():
    model = Sequential()
    model.add(Conv1D(48, 2, activation="relu", input_shape=(57,1)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(128,2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    #32
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',#sparse_categorical_crossentropy
      optimizer = "adam",
                metrics = [tf.keras.metrics.CategoricalAccuracy()])#CategoricalTruePositives(),CategoricalTrueNegative(),CategoricalFalseNegative()])#,CategoricalTrueNegative(),CategoricalFalseNegative(),CategoricalFalsePositive()]) #,metric_recall,metric_FPR
    #model.summary()
    return model
def create_model_html_full1():
    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(22,1))) #48
    model.add(Conv1D(32, 2, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,2, activation="relu"))
    model.add(Conv1D(64,2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',#sparse_categorical_crossentropy
      optimizer ="adam",
                metrics = [tf.keras.metrics.CategoricalAccuracy()])#CategoricalTruePositives(),CategoricalTrueNegative(),CategoricalFalseNegative()])#,CategoricalTrueNegative(),CategoricalFalseNegative(),CategoricalFalsePositive()]) #,metric_recall,metric_FPR
    return model
def create_model_url_full1():
    model = Sequential()
    model.add(Conv1D(48, 2, activation="relu", input_shape=(35,1)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(128,2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    #32
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',#sparse_categorical_crossentropy
      optimizer ="adam", #"",#opt,adam
                metrics = [tf.keras.metrics.CategoricalAccuracy()])#CategoricalTruePositives(),CategoricalTrueNegative(),CategoricalFalseNegative()])#,CategoricalTrueNegative(),CategoricalFalseNegative(),CategoricalFalsePositive()]) #,metric_recall,metric_FPR
    #model.summary()
    return model
#for custom wstphish PWD
def create_model_cnn_full():
    model = Sequential()
    model.add(Conv1D(48, 2, activation="relu", input_shape=(65,1)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(128,2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    #32
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',#sparse_categorical_crossentropy 
      optimizer = "adam",               
                metrics = [tf.keras.metrics.CategoricalAccuracy()])#CategoricalTruePositives(),CategoricalTrueNegative(),CategoricalFalseNegative()])#,CategoricalTrueNegative(),CategoricalFalseNegative(),CategoricalFalsePositive()]) #,metric_recall,metric_FPR
    #model.summary()
    return model 
def create_model_html_full():
    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(29,1))) #48
    model.add(Conv1D(32, 2, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,2, activation="relu"))
    model.add(Conv1D(64,2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',#sparse_categorical_crossentropy 
      optimizer ="adam",               
                metrics = [tf.keras.metrics.CategoricalAccuracy()])#CategoricalTruePositives(),CategoricalTrueNegative(),CategoricalFalseNegative()])#,CategoricalTrueNegative(),CategoricalFalseNegative(),CategoricalFalsePositive()]) #,metric_recall,metric_FPR
    return model 

def create_model_url_full():
    model = Sequential()
    model.add(Conv1D(48, 2, activation="relu", input_shape=(36,1)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(128,2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    #32
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',#sparse_categorical_crossentropy 
      optimizer ="adam", #"",#opt,adam               
                metrics = [tf.keras.metrics.CategoricalAccuracy()])#CategoricalTruePositives(),CategoricalTrueNegative(),CategoricalFalseNegative()])#,CategoricalTrueNegative(),CategoricalFalseNegative(),CategoricalFalsePositive()]) #,metric_recall,metric_FPR
    #model.summary()
    return model 

# for custom pwd
def time_metri(model,train_x,train_y,test_x,test_y):
    start=time.time()
    model.fit(train_x,train_y)
    end=time.time()
#     print('end is',end)
    tr_time=(end-start)#second
    
    pred=model.predict(test_x)
    te_time=time.time()-end
    print('te_time is %.4f'% te_time)
    print('tr_time is %.4f'%tr_time)
    tnr,tpr,f1_score=get_fpr(test_y,pred)
    
    print('tpr is %.2f, tnr is %.2f,f1 is %.2f'%(tpr,tnr,f1_score))
    return model,tr_time,te_time,tnr,tpr,f1_score 
# for pwd_laseto
def readHtmlFile(hpath): 
  
    try:
        with open(hpath) as f: # encoding='gbk'gb2312
            HTML=f.read()
            print('html success',hpath)
    except:  
        try:
            with open(hpath,encoding='gbk') as f: # encoding='gbk'gb2312
                HTML=f.read()
                print('html success',hpath)
        except:
            try:
                with open(hpath,encoding='gb2312') as f: # encoding='gbk'gb2312
                    HTML=f.read()
                    print('html success',hpath)
            except:
                print('no this file',hpath)
                return 'default'     
    return HTML            
def laseto(HTML):
    ''' check the language of the webpage'''
    soup = BeautifulSoup(HTML,"html.parser") 
    ht=soup.find('html') 
    try:    
        if ht['lang'] is not None:
            lan=ht['lang']
            print('language is',lan)
    except:
        if soup.find('title') :
            #print('title is',soup.find('title'))
            te=soup.find('title').contents
            if len(te)!=0:
                te=str(te[0])
                #print('te 0 is',te)
                resp=cld3.get_language(te)
                lan=resp[0]
                print('language is',lan)
            else:
                print('cannot judge language')
                lan='None'
        else:
            print('cannot judge language')
            lan='None'
    return lan
def sep_metri(model,test_x,test_y):
    pred=model.predict(test_x)
    print('test_y is %d, pred is %d'%(test_y,pred))
    if pred==test_y==1:
        tp=1
        tn=fp=fn=0
    elif pred==test_y==0:
        tn=1
        fp=tp=fn=0
    elif pred!=test_y and test_y==0:
        fp=1
        tp=tn=fn=0
    elif pred!=test_y and test_y==1:
        fn=1
        fp=tn=tp=0
    else:
        print('error')
        tp=tn=fp=fn=0
    return tp,tn,fp,fn
