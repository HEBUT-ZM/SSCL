# -*- coding: utf-8 -*-
import tensorflow as tf
import datetime
import numpy as np
import os
import scipy.io as sio
import copy
from L_Datagenerator import ImageDataGenerator
from UL_Datagenerator import UImageDataGenerator
import ResNet
slim = tf.contrib.slim
import getAllPath as GP

DATASET_DIR = "AID/"   # load data
CHECKPOINT_DIR = 'checkpoint_sscl/'
weight_path = "weights/"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CLASSES = 30
BATCHSIZE = 200
UBATCHSIZE = 200
LEARNINT_RATE = 0.001
EPOCHS =100
tr_rate=0.6
ltr_rate=0.1

x = tf.placeholder(tf.float32, shape=(None,256,256,3))
ux = tf.placeholder(tf.float32,shape=(None,256,256,3)) 
y = tf.placeholder(tf.int64, None)
is_training = tf.placeholder('bool', [])

def Cluster(centroids,features,BATCHSIZE,labels,NUM_CLASSES): 
        m=features.shape[0]
        clusterAssment = np.mat(np.zeros((m,2)))       
        mindistance = list(np.zeros((NUM_CLASSES,1)))
        unique_label=list(np.unique(labels))
        clusterChange = True                         
        while clusterChange:
             clusterChange = False           
             clusterAssment=clusterAssment.tolist()
             for i in range(BATCHSIZE):
                 distance = np.linalg.norm(centroids[labels[i],:]-features[i,:])
                 clusterAssment[i][0]=labels[i]
                 clusterAssment[i][1]=distance
             clusterAssment=np.mat(clusterAssment)
             for j in range(NUM_CLASSES):
                if j in unique_label:
                    pointsInCluster_l = clusterAssment[np.nonzero(clusterAssment[0:BATCHSIZE,0].A == j)[0]]
                    mindistance.append(np.max(pointsInCluster_l,axis=0)[0,0])
                else:
                    mindistance.append(0)          
             clusterAssment=clusterAssment.tolist()
             for i in range(BATCHSIZE,m):
                minDist = 100000.0
                minIndex = -1 
                distanceM=[]
                for j in range(NUM_CLASSES):
                    dis = np.linalg.norm(centroids[j,:] - features[i,:])
                    distanceM.append(dis)
                if min(distanceM)< mindistance[distanceM.index(min(distanceM))]:
                    minDist = min(distanceM)
                    minIndex = distanceM.index(min(distanceM))             
                if clusterAssment[i][0] != minIndex:
                    clusterChange = True
                    clusterAssment[i][0]=minIndex
                    clusterAssment[i][1]=minDist                    
             clusterAssment=np.mat(clusterAssment) 
             for j in range(NUM_CLASSES):
                if j in unique_label:
                    pointsInCluster = features[np.nonzero(clusterAssment[:,0].A == j)[0]]
                    centroids[j,:] = np.mean(pointsInCluster,axis=0)
        return centroids

def ucenterloss(labels,features,centers,BATCHSIZE):
        labels = tf.reshape(labels, [-1])
        labels=tf.cast(labels, tf.int32)       
        centers_batch = tf.gather(centers, labels)
        centerloss = tf.nn.l2_loss(features[0:BATCHSIZE,:] - centers_batch)
        return centerloss 

utrain_path=[]
train_path=[]
train_label = []
test_path=[]
test_label = []
#utrain_path,train_path,train_label,test_path,test_label=GP.get_original_Data(DATASET_DIR,tr_rate,ltr_rate)
#if not os.path.isdir('path'):      
#    os.mkdir('path')
#np.savez('path/AID',utrain_path,train_path,train_label,test_path,test_label)
path_label=np.load('path/AID.npz')
utrain_path=path_label['arr_0']
train_path=path_label['arr_1']
train_label=path_label['arr_2']
test_path=path_label['arr_3']
test_label=path_label['arr_4']

utr_data = UImageDataGenerator(
    images=utrain_path,
    batch_size=UBATCHSIZE,
    num_classes=NUM_CLASSES)
tr_data = ImageDataGenerator(
    images=train_path,
    labels=train_label,
    batch_size=BATCHSIZE,
    num_classes=NUM_CLASSES)
test_data = ImageDataGenerator(
    images=test_path,
    labels=test_label,
    batch_size=BATCHSIZE,
    num_classes=NUM_CLASSES,
    shuffle=False)

with tf.name_scope('input'):
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,tr_data.data.output_shapes)
    training_init_op=iterator.make_initializer(tr_data.data)
    test_init_op=iterator.make_initializer(test_data.data)
    next_batch = iterator.get_next()   
with tf.name_scope('uinput'):
    uiterator = tf.data.Iterator.from_structure(utr_data.data.output_types,utr_data.data.output_shapes)
    utraining_init_op=uiterator.make_initializer(utr_data.data)
    unext_batch = uiterator.get_next()   

with tf.name_scope("ResNet"): 
    depth = 50
    ResNetModel = ResNet.ResNetModel(x, ux, y, is_training, depth, NUM_CLASSES, BATCHSIZE)
    embedding=ResNetModel.embeddings
    center_update_op = ResNetModel.centers_update_op 
    center = ResNetModel.centers                          
    softmax_loss = ResNetModel.softmax_loss              
    acc = ResNetModel.accuracy                           
    pred = ResNetModel.predictions

with tf.name_scope("train"):
    train_layers = ['softmax','scale5']
    with tf.control_dependencies([center_update_op]):
         centroids=tf.py_func(Cluster,[center,embedding,BATCHSIZE,y,NUM_CLASSES],tf.float32)
         ucenter_loss=ucenterloss(y,embedding,centroids,BATCHSIZE)
         loss=softmax_loss+0.001*ucenter_loss
         update_op=tf.assign(center,centroids)
         train_op = ResNetModel.optimize(loss,learning_rate=LEARNINT_RATE, train_layers=train_layers)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    ResNetModel.load_original_weights(weight_path=weight_path, session=sess)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored .....")
    sess.graph.finalize()
    print("Training start:")
    utrain_batches_of_epoch = int(np.floor(utr_data.data_size / UBATCHSIZE))
    for epoch in range(EPOCHS):
        print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))
        sess.run(utraining_init_op)
        sess.run(training_init_op)                   
        for step in range(utrain_batches_of_epoch):
            if step%5 == 0:
               sess.run(training_init_op)
            img_batch, label_batch = sess.run(next_batch)
            uimg_batch = sess.run(unext_batch)            
            label_batch_number=copy.deepcopy(label_batch)
            label_batch_number=[np.argmax(one_hot)for one_hot in label_batch_number]      
            _,_,loss_value, acc_value = sess.run([update_op,train_op,loss,acc], feed_dict={x: img_batch,ux:uimg_batch,y:label_batch_number,is_training: True})               
            print("{} {} step loss = {:.4f}".format(datetime.datetime.now(), step, loss_value))
            print("{} {} step acc = {:.4f}".format(datetime.datetime.now(), step, acc_value))
        print("{} {} Start test".format(datetime.datetime.now(), epoch))
        test_acc = 0.0
        test_count = 0
        test_batches_of_epoch = int(np.floor(test_data.data_size / BATCHSIZE))  
        sess.run(test_init_op)
        for tag in range(test_batches_of_epoch):
            img_batch, label_batch = sess.run(next_batch)  
            label_batch_number=copy.deepcopy(label_batch)
            label_batch_number=[np.argmax(one_hot)for one_hot in label_batch_number]
            acc_value = sess.run(acc, feed_dict={x: img_batch,ux:img_batch,y:label_batch_number,is_training: False})           
            test_acc += acc_value
            test_count += 1
        test_acc /= test_count
        print("{} {} Test Accuracy = {:.4f}".format(datetime.datetime.now(), epoch, test_acc))       
        saver.save(sess, CHECKPOINT_DIR + str(epoch + 1) + '.ckpt')
