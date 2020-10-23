
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

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Model():
    def __init__(self, options):
        self.model = self.buildmodel(options)
        self.save_model_info()

        # 创建带池化层的model
    def buildmodel(self, options):
        print('[INFO]: Start to build model_1 with pool...')
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
        if not os.path.exists(options['savepath']):
            os.mkdir(options['savepath'])
        savename = 'model.h5'
        model.save_weights(os.path.join(options['savepath'], savename))
        return model
    def train(self,agent,options,batch_idx=0):
        self.model.load_weights('model/model.h5')
        Data_deque = deque()
        height = options['input_shape'][0]
        width = options['input_shape'][1]
        num_channels = options['input_shape'][2]
        # 小恐龙跳跃的概率
        start_prob_jump = options['start_prob_jump']
        end_prob_jump = options['end_prob_jump']
        interval_prob = options['interval_prob']
        # 如果需要继续训练，这里调整prob初始值
        prob=start_prob_jump
        # 操作num_operations后进行训练
        num_operations = options['num_operations']
        actions = np.zeros(options['num_actions'])
        actions[0]=1
        # (1, 0) -> do nothing
        # (0, 1) -> jump
        img, score, is_dead ,_= agent.frame_step(actions)
        size = (width, height)
        img = preprocess_img(img, size=size)
        # 模型训练的输入
        x_now = np.stack((img,) * num_channels, axis=2).reshape(1, height, width, num_channels)
        x_init = x_now
        i=0
        los=[]
        scre=[]
        rewd=[]
        re=0.0
        while True:
            i+=1
            actions = np.zeros(options['num_actions'])
            if random.random() <= prob:
                print('[INFO]: Dino actions randomly...')
                action_idx = random.randint(0, len(actions)-1)
                actions[action_idx] = 1
            else:
                print('[INFO]: Dino actions controlled by network...')
                Q_now = self.model.predict(x_now)
                action_idx = np.argmax(Q_now)
                actions[action_idx] = 1
            img, score, is_dead ,se= agent.frame_step(actions)
            img = preprocess_img(img, size=size)
            reward = self.score2reward(score, is_dead, actions)
            re+=reward
            img = img.reshape(1, height, width, 1)
            x_next = np.append(img, x_now[:, :, :, :num_channels-1], axis=3)
            Data_deque.append((x_now, action_idx, reward, x_next, is_dead, score))
            if len(Data_deque) > options['data_memory']:
                Data_deque.popleft()
            
            if i > num_operations:
                
                batch_idx += 1
                print('[INFO]: Start to train <Batch-%d>...' % batch_idx)
                loss = self.trainBatch(random.sample(Data_deque, options['batch_size']), options)
                los.append(loss)
                print('\t<Loss>: %.3f, <Action>: %d' % (loss, action_idx))
                if batch_idx % options['save_interval'] == 0:
                    if not os.path.exists(options['savepath']):
                        os.mkdir(options['savepath'])
                    
                    savename = options['savename'] + '_' + str(batch_idx) + '.h5'
                    self.model.save_weights(os.path.join(options['savepath'], savename))
                    save_dict(los, options['log_dir'], 'loss.pkl')
                if batch_idx == options['max_batch']:
                    break
            x_now = x_init if is_dead else x_next
            if is_dead:
                rewd.append(re)
                scre.append(se)
                re=0.0
                save_dict(rewd, options['log_dir'], 'reward.pkl')
                save_dict(scre, options['log_dir'], 'score.pkl')
            # 逐渐减小人为设定的控制，让网络自己决定如何行动
            if prob > end_prob_jump and i > num_operations:
                prob -= (start_prob_jump - end_prob_jump) / interval_prob
        savename = options['savename'] + '_' + str(batch_idx) + '.h5'
        self.model.save_weights(os.path.join(options['savepath'], savename))

    # 训练一个Batch数据
    def trainBatch(self, data_batch, options):
        height = options['input_shape'][0]
        width = options['input_shape'][1]
        num_channels = options['input_shape'][2]
        inputs = np.zeros((options['batch_size'], height, width, num_channels))
        targets = np.zeros((inputs.shape[0], options['num_actions']))
        for i in range(len(data_batch)):
            x_now, action_idx, reward, x_next, is_dead, _ = data_batch[i]
            inputs[i: i+1] = x_now
            targets[i] = self.model.predict(x_now)
            Q_next = self.model.predict(x_next)
            if is_dead:
                targets[i, action_idx] = reward
            else:
                targets[i, action_idx] = reward + options['rd_gamma'] * np.max(Q_next)
        loss = self.model.train_on_batch(inputs, targets)
        return loss
    # scrore转reward
    def score2reward(self, score, is_dead, actions):
        
        reward = 0.1
        if is_dead:
            reward = -1
        return reward
    # 导入权重
    def load_weight(self, weight_path):
        self.model.load_weights(weight_path)
    # 保存模型信息
    def save_model_info(self, savename='model.json'):
        with open(savename, 'w') as f:
            json.dump(self.model.to_json(), f)
    def __repr__(self):
        return '[Model]:\n%s' % self.model
    def __str__(self):
        return '[INFO]: model_1-CNN built by keras...'
