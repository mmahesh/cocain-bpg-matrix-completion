import time
st_time = time.time()
time_vals = [st_time]

import numpy as np
from scipy.sparse import rand as sprand
import scipy.sparse as sp

import torch
from torch.autograd import Variable
# np.random.seed(0)
import torch.nn.functional as F

torch.manual_seed(2)

import argparse
parser = argparse.ArgumentParser(description='Movielens')
parser.add_argument('--lam', '--regularization-parameter', default=1e-1,type=float,  dest='lam')
parser.add_argument('--algo', '--algorithm', default=1,type=int,  dest='algo')
parser.add_argument('--beta', '--palm-beta', default=0,type=float,  dest='beta')
parser.add_argument('--max_epochs', '--max-epochs', default=1000,type=int,  dest='max_epochs')
parser.add_argument('--data_option', '--data-option', default=2,type=int,  dest='data_option')
parser.add_argument('--num_factors', '--num-factors', default=5,type=int,  dest='num_factors')
parser.add_argument('--fun_num', '--fun_num', default=1,type=int,  dest='fun_num')

args = parser.parse_args()
lam = args.lam

data_option=args.data_option
if data_option ==1:
	n_users = 943
	n_items = 1682
	ratings = sp.load_npz('data/ml-100k-train.npz')
	test_ratings = sp.load_npz('data/ml-100k-test.npz')
elif data_option==2:
	n_users = 6040
	n_items = 3706
	ratings = sp.load_npz('data/ml-1m-train.npz')
	test_ratings = sp.load_npz('data/ml-1m-test.npz')
else:
	n_users = 69878
	n_items = 10677
	ratings = sp.load_npz('data/ml-10m-train.npz')
	test_ratings = sp.load_npz('data/ml-10m-test.npz')



class MatrixFactorization(torch.nn.Module):
	#https://www.ethanrosenthal.com/2017/06/20/matrix-factorization-in-pytorch/
	def __init__(self, n_users, n_items, n_factors=10):
		super(MatrixFactorization, self).__init__()
		self.user_factors = torch.nn.Embedding(n_users, 
												 n_factors,
												 sparse=True)
		self.item_factors = torch.nn.Embedding(n_items, 
												 n_factors,
												 sparse=True)
		
	def forward(self, user, item):
		return (self.user_factors(user) * self.item_factors(item)).sum(1)


def loss_func1(output, target,model):
	loss = torch.sum((output - target)**2)*0.5
	return loss

def loss_func(output, target,temp_model):
	loss = torch.sum((output - target)**2)*0.5
	temp1_loss = 0
	for param in temp_model.parameters():
		temp1_loss = temp1_loss + (param.norm(2)**2)*0.5*lam
	return loss+ temp1_loss

def loss_func2(output, target,temp_model):
	loss = torch.sum((output - target)**2)*0.5
	temp1_loss = 0
	for param in temp_model.parameters():
		temp1_loss = temp1_loss + lam*torch.sum(torch.abs(param))
	return loss+ temp1_loss

def grad(A, U, Z, lam, fun_num=1):
	model_copy = MatrixFactorization(n_users, n_items, n_factors=n_factors)
	model_copy.double()

	count = 0
	for param in model_copy.parameters():
		if count==0:
			param.data = U.data
			count+=1
		else:
			param.data = Z.data

	if fun_num in [1,2]:
		predictions = (model_copy(index_set_rows,index_set_cols))
		loss_temp = loss_func1(predictions, ratings, model_copy)
		loss_temp.backward()
		count = 0
		for param in model_copy.parameters():
			if count == 0:
				G0 = param.grad.data.to_dense()
				param.grad.data.zero_()
				count+=1
			else:
				G1 = param.grad.data.to_dense()
				param.grad.data.zero_()
		del model_copy

		return G0,G1


def breg( U, Z, U1, Z1, breg_num=1, c_1=1,c_2=1):
	if breg_num==1:

		grad_h_1_a = U1*((U1.norm(2)**2) + (Z1.norm(2)**2))
		grad_h_1_b = Z1*((U1.norm(2)**2) + (Z1.norm(2)**2))
		grad_h_2_a = U1
		grad_h_2_b = Z1
		
		temp_1 =  (0.25*(((U.norm(2)**2) + (Z.norm(2)**2))**2)) - (0.25*(((U1.norm(2)**2) + (Z1.norm(2)**2))**2))\
		-torch.sum((U-U1)*grad_h_1_a) -torch.sum((Z-Z1)*grad_h_1_b)
		temp_2 = (0.5*((U.norm(2)**2) + (Z.norm(2)**2))) - (0.5*(((U1.norm(2)**2) + (Z1.norm(2)**2))))\
		-torch.sum((U-U1)*grad_h_2_a) -torch.sum((Z-Z1)*grad_h_2_b)
		if abs(temp_1) <= 1e-10:
			temp_1 = 0
		if abs(temp_2) <= 1e-10:
			temp_2 = 0
		if c_1*temp_1 + c_2*temp_2 <0:
			return 0
		else:
			return c_1*temp_1 + c_2*temp_2

def main_func(A, U, Z, lam, fun_num=1,option=1):
	model_copy = MatrixFactorization(n_users, n_items, n_factors=n_factors)
	model_copy.double()
	count = 0
	for param in model_copy.parameters():
		if count==0:
			param.data = U.data
			count+=1
		else:
			param.data = Z.data

	if fun_num==1:
		temp_predictions = (model_copy(index_set_rows,index_set_cols))
		if option==1:
			t_loss = loss_func(temp_predictions, ratings, model_copy)
			loss_val= t_loss
			del model_copy
			return loss_val
		elif option==2:
			temp_predictions = (model_copy(index_set_rows,index_set_cols))
			
			t_loss = loss_func1(temp_predictions, ratings, model_copy)
			loss_val= t_loss
			del model_copy
			return loss_val
		else:
			raise
	if fun_num==2:
		temp_predictions = (model_copy(index_set_rows,index_set_cols))
		if option==1:
			t_loss = loss_func2(temp_predictions, ratings, model_copy)
			loss_val= t_loss
			del model_copy
			return loss_val
		elif option==2:
			temp_predictions = (model_copy(index_set_rows,index_set_cols))
			
			t_loss = loss_func1(temp_predictions, ratings, model_copy)
			loss_val= t_loss
			del model_copy
			return loss_val
		else:
			raise





def abs_func(A, U, Z, U1, Z1, lam, abs_fun_num=1, fun_num=1):
	if abs_fun_num == 1:

		G0,G1 = grad(A, U1, Z1, lam, fun_num=fun_num)
		temp_1 = (lam*0.5*(U.norm(2)**2))
		temp_2 = (lam*0.5*(Z.norm(2)**2))

		return  temp_1 + temp_2 + main_func(A, U1, Z1, lam, fun_num=fun_num, option=2) + torch.sum(torch.mul((U-U1),G0)) + torch.sum(torch.mul((Z-Z1),G1))
	if abs_fun_num == 2:

		G0,G1 = grad(A, U1, Z1, lam, fun_num=fun_num)
		temp_1 = lam*torch.sum(torch.abs(U))
		temp_2 = lam*torch.sum(torch.abs(Z))

		return  temp_1 + temp_2 + main_func(A, U1, Z1, lam, fun_num=fun_num, option=2) + torch.sum(torch.mul((U-U1),G0)) + torch.sum(torch.mul((Z-Z1),G1))


def find_gamma(A,U,Z,prev_U,prev_Z,uL_est, lL_est):
	gamma = 1
	kappa = (0.99)*(uL_est/(uL_est+lL_est))
	y_U = U+ gamma*(U-prev_U)
	y_Z = Z+ gamma*(Z-prev_Z)

	while ((kappa*breg(prev_U, prev_Z, U, Z, breg_num=breg_num,c_1=c_1,c_2=c_2)-breg(U, Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2))<-1e-5):
		gamma = gamma*0.9
		y_U = U+ gamma*(U-prev_U)
		y_Z = Z+ gamma*(Z-prev_Z)
		
		if gamma <= 1e-5:
			gamma = 0

	return y_U,y_Z, gamma

def do_lb_search(A, U, Z, U1, Z1, lam, uL_est,lL_est):
	y_U,y_Z, gamma = find_gamma(A,U,Z,U1,Z1,uL_est, lL_est)
	while((abs_func(A, U, Z, y_U, y_Z, lam, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func(A, U, Z, lam, fun_num=fun_num)\
		-(lL_est*breg(U, Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2)))>1e-5):
		lL_est = (1.1)*lL_est
		print('lL_est '+ str(lL_est))
		y_U,y_Z, gamma = find_gamma(A,U,Z,U1,Z1,uL_est, lL_est)
	return lL_est, y_U, y_Z, gamma

def make_update(U1, Z1,uL_est=1,lam=0,fun_num=1, abs_fun_num=1,breg_num=1, A=1):
 
	if breg_num ==1:

		grad_u, grad_z = grad(A, U1, Z1, lam, fun_num=fun_num)

		if abs_fun_num == 1:
			temp_const = c_1*(U1.norm(2)**2 + Z1.norm(2)**2).item() +c_2
			
			p_l =  (U1 - (1/(uL_est*temp_const))*grad_u)
			q_l =  (Z1 - (1/(uL_est*temp_const))*grad_z)
			
			# compute q_lambda
			coeff = [c_1*((p_l.norm(2)**2).item() + (q_l.norm(2)**2).item())*temp_const*temp_const, 0,(c_2 + (lam/uL_est)), -1]
			temp_y = np.roots(coeff)[-1].real
			return temp_const*temp_y*p_l, temp_const*temp_y*q_l
		if abs_fun_num == 2:
			temp_const = (c_1*(U1.norm(2)**2 + Z1.norm(2)**2).item()) +c_2
			tp_l = (1/uL_est)*grad_u - temp_const*U1
			p_l = F.relu(torch.abs(-tp_l)-(lam/uL_est))*(-tp_l).sign()
			tq_l = (1/uL_est)*grad_z - temp_const*Z1
			q_l = F.relu(torch.abs(-tq_l)-(lam/uL_est))*(-tq_l).sign()
			coeff = [c_1*((p_l.norm(2)**2).item() + (q_l.norm(2)**2).item()), 0,c_2, -1]
			temp_y = np.roots(coeff)[-1].real
			return temp_y*p_l, temp_y*q_l


def do_ub_search(A, y_U,y_Z, uL_est):
	x_U,x_Z = make_update(y_U,y_Z, uL_est,lam, fun_num=fun_num, abs_fun_num=abs_fun_num,breg_num=breg_num, A=A)

	while((abs_func(A, x_U,x_Z,y_U,y_Z, lam, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func(A, x_U,x_Z, lam,  fun_num=fun_num)\
		+(uL_est*breg(x_U, x_Z, y_U, y_Z, breg_num=breg_num,c_1=c_1,c_2=c_2)))<-1e-5):
		
		uL_est = (1.1)*uL_est
		print('uL_est is '+ str(uL_est))

		x_U,x_Z = make_update(y_U,y_Z, uL_est,lam, fun_num=fun_num, abs_fun_num=abs_fun_num,breg_num=breg_num, A=A)
	return uL_est, x_U, x_Z

def rmse_fun(output, target):
	output.double()
	target.double()
	loss = torch.sqrt(torch.mean((output - target)**2))
	return loss
n_factors = args.num_factors
model = MatrixFactorization(n_users, n_items, n_factors=n_factors)
model.double()

for param in model.parameters():
	param.data = torch.nn.init.ones_(param.data)*0.5
# 	# 0.1 for ml-100k 
# 	# 0.5 for 1m
# but same initialization for all algorithms

rows, cols = ratings.nonzero()
index_set_rows = Variable(torch.LongTensor(rows))
index_set_cols = Variable(torch.LongTensor(cols))
ratings = Variable(torch.DoubleTensor(ratings.data))

test_rows,test_cols = test_ratings.nonzero()
test_index_set_rows = Variable(torch.LongTensor(test_rows))
test_index_set_cols = Variable(torch.LongTensor(test_cols))
test_ratings = Variable(torch.DoubleTensor(test_ratings.data))

algo = args.algo # 1) bpg 2) iPALM
from optimizers_pytorch import *
if algo==1:
	max_val = np.linalg.norm(ratings.data)
	pass

if algo==2:
	beta = args.beta
	fun_num=args.fun_num
if algo in [3,4]:
	max_val = np.linalg.norm(ratings.data)
	import torch.optim as optim
	optimizer= BPG_MF(model.parameters(), max_val=max_val,lam=lam)
	A = ratings
	uL_est = 1e-1
	lL_est_main = 1e-1
	lL_est = lL_est_main
if algo in [1,3, 4]:
	A = ratings
	abs_fun_num=args.fun_num
	fun_num=args.fun_num
	breg_num=1
	c_1 = 3
	c_2 = max_val

max_epochs = args.max_epochs
func_vals = []
train_rmse_vals = []
test_rmse_vals = []
for epoch in range(max_epochs):
	if algo ==1:
		U = model.user_factors.state_dict()['weight'].data
		Z = model.item_factors.state_dict()['weight'].data
		
		if epoch==0:
			predictions = (model(index_set_rows,index_set_cols))
			if fun_num==1:
				func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
			elif fun_num==2:
				func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
			else:
				raise
			train_rmse = rmse_fun(predictions, ratings)
			train_rmse_vals = train_rmse_vals + [train_rmse.item()]
			test_predictions = (model(test_index_set_rows,test_index_set_cols))
			test_rmse = rmse_fun(test_predictions, test_ratings)
			test_rmse_vals = test_rmse_vals + [test_rmse.item()]

		U,Z = make_update(U,Z, 1.1,lam, fun_num=fun_num, abs_fun_num=abs_fun_num,breg_num=breg_num, A=A)


		count = 0
		for param in model.parameters():
			if count==0:
				param.data = U.data
				count+=1
			else:
				param.data = Z.data

		predictions = (model(index_set_rows,index_set_cols))
		if fun_num==1:
			loss = loss_func(predictions, ratings, model)
		else:
			loss = loss_func2(predictions, ratings, model)
		print(loss)


		if fun_num==1:
			func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
		elif fun_num==2:
			func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
		else:
			pass

		

		train_rmse = rmse_fun(predictions, ratings)
		train_rmse_vals = train_rmse_vals + [train_rmse.item()]

		test_predictions = (model(test_index_set_rows,test_index_set_cols))
		test_rmse = rmse_fun(test_predictions, test_ratings)
		test_rmse_vals = test_rmse_vals + [test_rmse.item()]

	
	if algo==2:
		predictions = (model(index_set_rows,index_set_cols))

		if epoch==0:
			if fun_num==1:
				func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
			elif fun_num==2:
				func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
			else:
				raise
			train_rmse = rmse_fun(predictions, ratings)
			train_rmse_vals = train_rmse_vals + [train_rmse.item()]
			test_predictions = (model(test_index_set_rows,test_index_set_cols))
			test_rmse = rmse_fun(test_predictions, test_ratings)
			test_rmse_vals = test_rmse_vals + [test_rmse.item()]


		temp_mat = (model.item_factors.state_dict()['weight'])
		L_1 = (torch.matmul(torch.t(temp_mat),temp_mat).norm(2)*1.1).item()
		if beta>0:
			step_size = (1/L_1)*2*(1-beta)/(1+ 2*beta)
		else:
			step_size = (1/(1.1*L_1))

		if epoch==0:
			prev_user_fact_mat = (model.user_factors.state_dict()['weight'])
		prev_user_fact_mat1 = (model.user_factors.state_dict()['weight'])
		count = 0
		for param in model.parameters():
			if count==0:
				param.data = param.data + beta*(param.data - prev_user_fact_mat.data)
				count+=1
			else:
				pass

		grad_u, grad_z = grad(ratings, model.user_factors.state_dict()['weight'].data, model.item_factors.state_dict()['weight'].data, lam, fun_num=1)

		count = 0
		for param in model.parameters():
			if count==0:
				if fun_num==1:
					param.data = (param.data - step_size*grad_u.data)/(1+(lam*step_size))
					count+=1
				elif fun_num==2:
					temp_vec= param.data - step_size*grad_u.data
					param.data = F.relu(torch.abs(temp_vec) - lam*step_size)*(temp_vec.sign())
					count+=1
				else:
					pass
			else:
				pass
				
	
		prev_user_fact_mat = prev_user_fact_mat1
		# optimizer.zero_grad()


		# Step 2
		predictions = (model(index_set_rows,index_set_cols))
		
		
		temp_mat = (model.user_factors.state_dict()['weight'])
		L_2 = (torch.matmul(torch.t(temp_mat),temp_mat).norm(2)*1.1).item()
		if beta>0:
			step_size = (1/L_2)*2*(1-beta)/(1+ 2*beta)
		else:
			step_size = (1/(1.1*L_1))

		
		if epoch==0:
			prev_item_fact_mat = (model.item_factors.state_dict()['weight'])
		prev_item_fact_mat1 = (model.item_factors.state_dict()['weight'])

		count = 0
		for param in model.parameters():
			if count==0:
				count+=1
			else:
				param.data = param.data + beta*(param.data - prev_item_fact_mat.data)
		

		grad_u, grad_z = grad(ratings, model.user_factors.state_dict()['weight'].data, model.item_factors.state_dict()['weight'].data, lam, fun_num=1)

		count = 0
		for param in model.parameters():
			if count==0:
				count+=1
			else:
				if fun_num==1:
					param.data = (param.data - step_size*grad_z.data)/(1+(lam*step_size))
				elif fun_num==2:
					temp_vec= param.data - step_size*grad_z.data
					param.data = F.relu(torch.abs(temp_vec) - lam*step_size)*(temp_vec.sign())
					count+=1
				else:
					pass


		prev_item_fact_mat = prev_item_fact_mat1
		

		predictions = (model(index_set_rows,index_set_cols))

		if fun_num==1:
			loss1 = loss_func(predictions, ratings, model)
		elif fun_num==2:
			loss1 = loss_func2(predictions, ratings, model)
		else:
			pass
		print('loss '+ str(loss1))
		if fun_num==1:
			func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
		elif fun_num==2:
			func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
		else:
			pass

		train_rmse = rmse_fun(predictions, ratings)
		train_rmse_vals = train_rmse_vals + [train_rmse.item()]

		test_predictions = (model(test_index_set_rows,test_index_set_cols))
		test_rmse = rmse_fun(test_predictions, test_ratings)
		test_rmse_vals = test_rmse_vals + [test_rmse.item()]

	if algo ==3:
		

		U = model.user_factors.state_dict()['weight'].data
		Z = model.item_factors.state_dict()['weight'].data
		# print(epoch)
		if epoch == 0:
			prev_U = U.clone()
			prev_Z = Z.clone()
			print('cloned')
		if epoch==0:
			predictions = (model(index_set_rows,index_set_cols))
			if fun_num==1:
				func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
			elif fun_num==2:
				func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
			else:
				raise
			train_rmse = rmse_fun(predictions, ratings)
			train_rmse_vals = train_rmse_vals + [train_rmse.item()]
			test_predictions = (model(test_index_set_rows,test_index_set_cols))
			test_rmse = rmse_fun(test_predictions, test_ratings)
			test_rmse_vals = test_rmse_vals + [test_rmse.item()]
		
		lL_est, y_U, y_Z, gamma = do_lb_search(A, U, Z, prev_U, prev_Z, lam, uL_est,lL_est=lL_est_main)
		prev_U = U
		prev_Z = Z
		uL_est, U, Z = do_ub_search(A, y_U,y_Z, uL_est)

		count = 0
		for param in model.parameters():
			if count==0:
				param.data = U.data
				count+=1
			else:
				param.data = Z.data

		predictions = (model(index_set_rows,index_set_cols))

		if fun_num==1:
			loss = loss_func(predictions, ratings, model)
		else:
			loss = loss_func2(predictions, ratings, model)

		print(loss)

		if fun_num==1:
			func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
		elif fun_num==2:
			func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
		else:
			pass
		

		train_rmse = rmse_fun(predictions, ratings)
		train_rmse_vals = train_rmse_vals + [train_rmse.item()]

		test_predictions = (model(test_index_set_rows,test_index_set_cols))
		test_rmse = rmse_fun(test_predictions, test_ratings)
		test_rmse_vals = test_rmse_vals + [test_rmse.item()]

	if algo ==4:
		

		U = model.user_factors.state_dict()['weight'].data
		Z = model.item_factors.state_dict()['weight'].data
		if epoch == 0:
			prev_U = U.clone()
			prev_Z = Z.clone()
			print('cloned')
		if epoch==0:
			predictions = (model(index_set_rows,index_set_cols))
			if fun_num==1:
				func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
			elif fun_num==2:
				func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
			else:
				raise
			train_rmse = rmse_fun(predictions, ratings)
			train_rmse_vals = train_rmse_vals + [train_rmse.item()]
			test_predictions = (model(test_index_set_rows,test_index_set_cols))
			test_rmse = rmse_fun(test_predictions, test_ratings)
			test_rmse_vals = test_rmse_vals + [test_rmse.item()]
		

		uL_est, U, Z = do_ub_search(A, U,Z, uL_est)

		count = 0
		for param in model.parameters():
			if count==0:
				param.data = U.data
				count+=1
			else:
				param.data = Z.data


		predictions = (model(index_set_rows,index_set_cols))

		if fun_num==1:
			loss = loss_func(predictions, ratings, model)
		else:
			loss = loss_func2(predictions, ratings, model)

		print(loss)

		if fun_num==1:
			func_vals = func_vals + [loss_func(predictions, ratings, model).item()]
		elif fun_num==2:
			func_vals = func_vals + [loss_func2(predictions, ratings, model).item()]
		else:
			pass
		

		train_rmse = rmse_fun(predictions, ratings)
		train_rmse_vals = train_rmse_vals + [train_rmse.item()]

		test_predictions = (model(test_index_set_rows,test_index_set_cols))
		test_rmse = rmse_fun(test_predictions, test_ratings)
		test_rmse_vals = test_rmse_vals + [test_rmse.item()]

	time_vals = time_vals + [time.time()]


abs_fun_num= args.fun_num
breg_num=1
if algo==1:
	filename = 'fun_results/bpg_mf_movielens_fun_name_'+str(fun_num)+'_dataset_option_'\
		+str(data_option)+'_abs_fun_num_'+str(abs_fun_num)\
		+'_breg_num_'+str(breg_num) + '_lam_val_'+str(lam)+'.txt'
	np.savetxt(filename,np.c_[func_vals, time_vals, train_rmse_vals, test_rmse_vals])
elif algo==2:
	filename = 'fun_results/palm_mf_movielens_fun_name_'+str(fun_num)+'_dataset_option_'+str(data_option)\
		+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)\
		+'_beta_'+str(beta)+ '_lam_val_'+str(lam)+'.txt'
	np.savetxt(filename,np.c_[func_vals, time_vals, train_rmse_vals, test_rmse_vals])
elif algo==3:
	filename = 'fun_results/cocain_mf_movielens_fun_name_'+str(fun_num)+'_dataset_option_'+str(data_option)\
			+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ '_lam_val_'+str(lam)+'.txt'
	np.savetxt(filename,np.c_[func_vals, time_vals, train_rmse_vals, test_rmse_vals])
elif algo==4:
	filename = 'fun_results/bpg_mf_wb_movielens_fun_name_'+str(fun_num)+'_dataset_option_'+str(data_option)\
			+'_abs_fun_num_'+str(abs_fun_num)+'_breg_num_'+str(breg_num)+ '_lam_val_'+str(lam)+'.txt'
	np.savetxt(filename,np.c_[func_vals, time_vals, train_rmse_vals, test_rmse_vals])
else:
	raise

