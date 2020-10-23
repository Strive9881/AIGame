
import os
import json
import keras
import random
from collections import deque
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Activation, Conv2D, Flatten, Dense, MaxPooling2D,Input
import sys
sys.path.append('..')
from utils.utils import *


class Model():
    def __init__(self, options):
        self.model = self.buildmodel(options)
    def buildmodel(self,options):
        inputs = Input(shape=options['input_shape'])
        conv_1_out = Conv2D(32, (3, 3), activation='relu',padding='same', name='block1_conv1')(inputs)
        conv_1_out = Conv2D(32, (3, 3), activation='relu',padding='same', name='block1_conv2')(conv_1_out)
        ds_conv_1_out = MaxPooling2D((2, 2),strides=(2, 2),name='block1_pool')(conv_1_out)
        #ds_conv_1_out=Activation('relu')(ds_conv_1_out)
        
        conv_2_out = Conv2D(32, (3, 3), activation='relu',padding='same', name='block2_conv1')(ds_conv_1_out)
        conv_2_out = Conv2D(32, (3, 3), activation='sigmoid',padding='same', name='block2_conv2')(conv_2_out)
        ds_conv_2_out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_out)
        #ds_conv_2_out=Activation('relu')(ds_conv_2_out)
        
        conv_3_out = Conv2D(64, (3, 3), activation='relu',padding='same', name='block3_conv1')(ds_conv_2_out)
        conv_3_out = Conv2D(64, (3, 3), activation='relu',padding='same', name='block3_conv2')(conv_3_out)
        conv_3_out = Conv2D(64, (3, 3), activation='relu',padding='same', name='block3_conv3')(conv_3_out)
        ds_conv_3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv_3_out)
        #ds_conv_3_out=Activation('relu')(ds_conv_3_out)
        
        conv_4_out = Conv2D(64, (3, 3), activation='relu',padding='same', name='block4_conv1')(ds_conv_3_out)
        conv_4_out = Conv2D(64, (3, 3), activation='relu',padding='same', name='block4_conv2')(conv_4_out)
        conv_4_out = Conv2D(64, (3, 3), activation='sigmoid',padding='same', name='block4_conv3')(conv_4_out)
        ds_conv_4_out = MaxPooling2D((2, 2), strides=(2, 2),name='block4_pool')(conv_4_out)
        #ds_conv_4_out=Activation('relu')(ds_conv_4_out)
        
        ds_conv_4_out = Flatten()(ds_conv_4_out)
        y = Dense(512 , activation='relu')(ds_conv_4_out)
        y = Dense(256 , activation='relu')(y)
        y = Dense(options['num_actions'])(y)

        model = KerasModel(inputs = inputs , outputs = y)
        model.compile(optimizer=Adam(lr=options['lr']) ,loss='mse')
        return model

    def train(self,agent,options,batch_idx=0):
        #self.model.load_weights('model/model.h5')
        height = options['input_shape'][0]
        width = options['input_shape'][1]
        num_channels=options['input_shape'][2]
        actions = np.zeros(options['num_actions'])
        actions[0]=1
        # (1, 0) -> do nothing
        # (0, 1) -> jump
        img, score, is_dead,_ = agent.frame_step(actions)
        size = (width, height)
        img = preprocess_img(img, size=size)
        # 模型训练的输入
        x_now = np.stack((img,) * num_channels, axis=2).reshape(1, height, width, num_channels)
        x_init = x_now
        sc=[]
        Q=0.0
        qc=0
        i=0
        while True:
            actions = np.zeros(options['num_actions'])
            Q_now = self.model.predict(x_now)
            Q+=np.max(Q_now)
            qc+=1
            action_idx = np.argmax(Q_now)
            actions[action_idx] = 1
            img, score, is_dead ,se= agent.frame_step(actions)
            if is_dead:
                sc.append(se)
                i+=1
            img = preprocess_img(img, size=size)
            img = img.reshape(1, height, width, 1)
            x_next = np.append(img, x_now[:, :, :, :num_channels-1], axis=3)
            x_now = x_init if is_dead else x_next
            if i==3:
                break
            if se>20000:
                break
        print(sc)
        return 1.0*sum(sc)/len(sc),1.0*Q/qc

    # 导入权重
    def load_weight(self, weight_path):
        self.model.load_weights(weight_path)