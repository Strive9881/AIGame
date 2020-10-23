import sys
from auto_nets.nets import *
from utils.utils import *
from game.trex_utils import *
from game.wrapped_trex import *

options = {
				"lr": 1e-3,
				"batch_size": 32,
				"max_batch": 150000,
				"input_shape": (96, 96, 4),
				"num_actions": 2,
				"savename": "autodino",
				"savepath": "./model",
				"log_dir": "logger",
				"data_dir": "data",
				"start_prob_jump": 0,
				"end_prob_jump": 0,
				"interval_prob": 1e5,
				"save_interval": 500,
				"num_operations": 200,
				"data_memory": 50000,
				"rd_gamma": 0.99,
				"use_canny": False,
				"num_sampling": 200,
			}


'''
Function:
	自动玩T-Rex Rush
Input:
	-model: 模型
	-model_type: 模型搭建的框架
'''
def auto_play(agent,model,modepath):
		model.load_weight(modepath)
		return model.train(agent, options)
		

if __name__ == '__main__':
    agent = GameState()
    model = Model(options)
    score=[]
    Q_Value=[]
    for i in range(96,121):
        path='model/dino_'+str(1000*i)+'.h5'
        print(path)
        sc,q=auto_play(agent,model,path)
        score.append(sc)
        Q_Value.append(q)
        save_dict(score, options['log_dir'], 'score.pkl')
        save_dict(Q_Value, options['log_dir'], 'Q_Value.pkl')
        