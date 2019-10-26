import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
import numpy as np

class BPG_MF(Optimizer):
   

	def __init__(self, params, lr=0.1, c_1=0, c_2=0,
				 lam=0, max_val=1):
	   
		defaults = dict(lr=lr, c_1=c_1, c_2=c_2,lam=lam,max_val=max_val)
		
		super(BPG_MF, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(BPG_MF, self).__setstate__(state)
		
	def step(self, closure=None):
		
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			c_1 = group['c_1']
			c_2 = group['c_2']
			lam = group['lam']
			lr = group['lr']
			max_val = group['max_val']

			# ||U||^2 + ||Z||^2
			total_norm = 0
			for param in group['params']:
				param_norm =  param.data.norm(2)**2
				total_norm += param_norm.item()
			step_size = 1/(1.1)
			c_1 = 3
			c_2 = max_val 
			# print(c_2)
			# print(c_2)
			temp_const = (c_1*(total_norm) + c_2)

			for param in group['params']:
				param.data = ((param.data - ((step_size/temp_const)*param.grad)))

			total_norm1 = 0
			for param in group['params']:
				param_norm =  param.data.norm(2)**2
				total_norm1 += param_norm.item()
			total_norm1 = total_norm1*temp_const*temp_const
			coeff = [c_1*(total_norm1), 0,(c_2 + (lam/1.1)), -1]
			temp_y = np.roots(coeff)[-1].real


			for param in group['params']:
				param.data = temp_const*(temp_y)*param.data

		return loss


