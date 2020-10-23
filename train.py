import sys
from nets.nets import *
from utils.utils import *
from game.trex_utils import *
from game.wrapped_trex import *
options = {
				"lr": 1e-4,
				"batch_size": 32,
				"max_batch": 130000,
				"input_shape": (96, 96,4),
				"num_actions": 2,
				"savename": "dino",
				"savepath": "model",
				"log_dir": "logger",
				"data_dir": "data",
				"start_prob_jump": 0.1,
				"end_prob_jump": 1e-4,
				"interval_prob": 1e5,
				"save_interval": 1000,
				"num_operations":200,
				"data_memory": 100000,
				"rd_gamma": 0.99,
				"use_canny": False,
				"num_sampling": 200,
			}
# 训练函数
def train():
	agent = GameState()
	model = Model(options)
	model.train(agent, options)

if __name__ == '__main__':
	train()