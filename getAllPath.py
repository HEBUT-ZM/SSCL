# -*- coding: utf-8 -*-
import glob
import numpy as np
import random
import os
import math
def getIndex(tr_rate,ltr_rate,classNum):
    u_trIndex=[]
    trIndex=[]
    teIndex=[]
    totalList = [i for i in range(0,classNum)]
    random.shuffle(totalList)
    tr_num=int(classNum*tr_rate)
    ltr_num=int(ltr_rate*classNum)
    for i in range(0,ltr_num):
        trIndex.append(totalList[i])
    for i in range(ltr_num,math.ceil(tr_num)):
        u_trIndex.append(totalList[i])
    for i in range(tr_num,math.ceil(0.8*classNum)):
        teIndex.append(totalList[i])
    return u_trIndex,trIndex,teIndex
def get_original_Path(path,tr_rate,ltr_rate):
    u_trPath=[]
    trPath = []
    tePath = []
    classNum=0
    image_path = np.array(glob.glob(path + '/*.jpg')).tolist()
    classNum=len(image_path)    
    print('classNum='+str(len(image_path)))    
    u_trIndex,trIndex,teIndex=getIndex(tr_rate,ltr_rate,classNum)
    print(len(u_trIndex))    
    print(len(trIndex))
    print(len(teIndex))
    for i in range(0,len(u_trIndex)):
        u_trPath.append(image_path[u_trIndex[i]])
    for i in range(0,len(trIndex)):
        trPath.append(image_path[trIndex[i]])
    for i in range(0,len(teIndex)):
        tePath.append(image_path[teIndex[i]])        
    return u_trPath,trPath,tePath
def get_original_Data(path,tr_rate,ltr_rate):
    u_train_path=[]
    train_path=[]
    train_label=[]
    test_path=[]
    test_label=[]    
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    label_dir.sort()
    for index,folder in enumerate(label_dir):
        print(folder)
        u_tr,tr,te=get_original_Path(folder,tr_rate,ltr_rate)
        u_train_path=u_train_path+u_tr
        train_path=train_path+tr
        for i in range(0,len(tr)):
            train_label.append(index)
        test_path=test_path+te
        for i in range(0,len(te)):
            test_label.append(index)
    return u_train_path,train_path,train_label,test_path,test_label



