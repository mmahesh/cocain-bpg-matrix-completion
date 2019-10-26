import numpy as np
from scipy.sparse import rand as sprand
import scipy.sparse as sp
import torch
from torch.autograd import Variable
np.random.seed(0)
torch.manual_seed(2)


# Train = 50%, Validation=25%, Test=25%
# 5 such splits
# Cross-validation from 1e-5,1e-4,..,1e5

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Inspired from
# https://github.com/fuhailin/Recommender-System/blob/master/MovieLens%20Test.ipynb
def train_test_split2(fileName,type=1):
	header = ['user_id', 'item_id', 'rating', 'timestamp']
	if(type==1):
		df = pd.read_csv(fileName, sep=',',dtype={'userId':np.int32,'movieId':np.int32,'rating':np.float32})
		print(df)
	elif(type==3):
		df = pd.read_csv(
				os.path.join(os.path.dirname(__file__), 'data/MovieLens-100k/u.data'),
				sep='\t',
				engine="python",
				encoding="latin-1",
				names=['user_id', 'item_id', 'rating', 'timestamp'])
	else:
		df = pd.read_csv(fileName, sep='::', names=header,engine = 'python')
	
	try:
		# for others
		n_users = df.user_id.unique().shape[0]
		users = df.user_id.max()
		n_items = df.item_id.unique().shape[0]
		items = df.item_id.max()
	except:
		# for ml-20
		n_users = df.userId.unique().shape[0]
		users = df.userId.max()
		n_items = df.movieId.unique().shape[0]
		items = df.movieId.max()
		temp_list1 = df.userId.unique()

	print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
	print('The biggest ID of users = ' + str(users) + ' | The biggest ID of movies = ' + str(items))
	


	full_user_id_vals = (df.user_id.values)
	full_item_id_vals = (df.item_id.values)


	train_data, test_data = train_test_split(df, test_size=0.2)
	train_data = pd.DataFrame(train_data)
	test_data = pd.DataFrame(test_data)

	train_user_id_vals = (train_data.user_id.values)
	train_item_id_vals = (train_data.item_id.values)
	train_rating_vals = train_data.rating.values

	main_count = 0
	count = 0
	user_dict = {}
	for item in full_user_id_vals:
		try:
			temp = user_dict[item]
		except:
			user_dict[item] = count
			count+=1
		full_user_id_vals[main_count] = user_dict[item]
		main_count+=1

	main_count = 0
	count = 0
	item_dict = {}
	for item in full_item_id_vals:
		try:
			temp = item_dict[item]
		except:
			item_dict[item] = count
			count+=1
		full_item_id_vals[main_count] = item_dict[item]
		main_count+=1

	main_count = 0
	for item in train_user_id_vals:
		train_user_id_vals[main_count] = user_dict[item]
		main_count+=1

	main_count = 0
	for item in train_item_id_vals:
		train_item_id_vals[main_count] = item_dict[item]
		main_count+=1

	test_user_id_vals = (test_data.user_id.values)
	test_item_id_vals = (test_data.item_id.values)
	test_rating_vals = test_data.rating.values

	main_count = 0
	for item in test_user_id_vals:
		test_user_id_vals[main_count] = user_dict[item]
		main_count+=1

	main_count = 0
	for item in test_item_id_vals:
		test_item_id_vals[main_count] = item_dict[item]
		main_count+=1

	train_data_mat = sp.csr_matrix((train_rating_vals, (train_user_id_vals, train_item_id_vals)), \
		(n_users, n_items))
	test_data_mat = sp.csr_matrix((test_rating_vals, (test_user_id_vals, test_item_id_vals)), \
		(n_users, n_items))
	return train_data_mat, test_data_mat

for i in [1,2,3]:
	# creating all datasets
	data_option = i
	if data_option == 1:
		train_data_mat,test_data_mat = train_test_split2('data/MovieLens-100k/u.data',3)
		sp.save_npz('data/ml-100k-train.npz', train_data_mat)
		sp.save_npz('data/ml-100k-test.npz', test_data_mat)
	elif data_option==2:
		train_data_mat,test_data_mat = train_test_split2('data/MovieLens-1M/ratings.dat',2)
		sp.save_npz('data/ml-1m-train.npz', train_data_mat)
		sp.save_npz('data/ml-1m-test.npz', test_data_mat)
	else:
		train_data_mat,test_data_mat = train_test_split2('data/MovieLens-10M/ratings.dat',2)
		sp.save_npz('data/ml-10m-train.npz', train_data_mat)
		sp.save_npz('data/ml-10m-test.npz', test_data_mat)