import scipy.io as sio
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import os
import pandas as pd
import utils as utils
from sklearn import datasets


max_seq_length = 256

# class Data_Amazon(object):
# 	def __init__(self, path, batch_size):
#
# 		self.batch_size = batch_size
#
# 		split_folder = os.path.join(path, 'cold2')
#
# 		item_content_file = os.path.join(path, 'item_raw2.csv')
# 		self.item_content_file = item_content_file
# 		train_file = os.path.join(split_folder, 'train.csv')
# 		test_cold_file = os.path.join(split_folder, 'test.csv')
# 		test_cold_iid_file = os.path.join(split_folder, 'test_item_ids.csv')
# 		# self.item_raw_data_file = os.path.join(path, 'raw-data.csv')
#
# 		# load split
# 		train = pd.read_csv(train_file, delimiter=",", header=None, dtype=np.int32).values.ravel().view(
# 			dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32)])
# 		train_user_ids = np.unique(train['uid'])
# 		train_item_ids = np.unique(train['iid'])
# 		self.train_item_ids = train_item_ids
#
# 		train_user_ids_map = {user_id: i for i, user_id in enumerate(train_user_ids)}
# 		train_item_ids_map = {user_id: i for i, user_id in enumerate(train_item_ids)}
#
# 		_train_i_for_inf = [train_user_ids_map[_t[0]] for _t in train]
# 		_train_j_for_inf = [train_item_ids_map[_t[1]] for _t in train]
# 		self.R_train = sp.coo_matrix(
# 			(np.ones(len(_train_i_for_inf)),
# 			 (_train_i_for_inf, _train_j_for_inf)),
# 			shape=[len(train_user_ids), len(train_item_ids)]).tolil(copy=False)
#
# 		self.train_items = self.record_list(_train_j_for_inf, _train_i_for_inf)
# 		self.train_u = _train_i_for_inf
# 		self.train_i = _train_j_for_inf
#
# 		with open(test_cold_iid_file) as f:
# 			test_item_ids = [int(line) for line in f]
# 			self.test_item_ids = test_item_ids
# 			test_data = pd.read_csv(test_cold_file, delimiter=",", header=None, dtype=np.int32).values.ravel()
#
# 			test_data = test_data.view(
# 				dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32)])
#
# 			test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}
#
# 			_test_ij_for_inf = [(t[0], t[1]) for t in test_data if t[1] in test_item_ids_map]
# 			# test_user_ids
# 			test_user_ids = np.unique(test_data['uid'])
# 			# test_user_ids_map
# 			# test_user_ids_map = {user_id: i for i, user_id in enumerate(test_user_ids)}
# 			test_user_ids_map = train_user_ids_map
# 			print(len(_test_ij_for_inf))
# 			## no 14429 #####
# 			# _test_i_for_inf, _test_j_for_inf = [], []
# 			# for _t in _test_ij_for_inf:
# 			# 	if _t[0] in test_user_ids_map:
# 			# 		_test_i_for_inf.append(test_user_ids_map[_t[0]])
# 			# 		_test_j_for_inf.append(test_item_ids_map[_t[1]])
# 			_test_i_for_inf = [test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
# 			_test_j_for_inf = [test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
# 			self.R_test = sp.coo_matrix(
# 				(np.ones(len(_test_i_for_inf)),
# 				 (_test_i_for_inf, _test_j_for_inf)),
# 				shape=[len(train_user_ids), len(test_item_ids)]).tolil(copy=False)
#
# 			self.test_items = self.record_list(_test_j_for_inf, _test_i_for_inf)
# 			self.test_u = _test_i_for_inf
# 			self.test_users = self.record_list(_test_i_for_inf, _test_j_for_inf)
#
# 		# item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)
#
# 		# item_content = tfidf(item_content)
# 		item_content = self.item_raw_load(item_content_file)
# 		from sklearn.utils.extmath import randomized_svd
# 		u, s, _ = randomized_svd(item_content, n_components=300, n_iter=5)
# 		item_content = u * s
# 		_, item_content = utils.prep_standardize(item_content)
#
# 		if sp.issparse(item_content):
# 			item_feature = item_content.tolil(copy=False)
# 		else:
# 			item_feature = item_content
# 		# timer.toc('loaded item feature sparse matrix: %s' % (str(item_content.shape))).tic()
# 		self.item_content = item_feature
#
#
# 		self.S_tr = item_feature[train_item_ids, :]
# 		self.S_te = item_feature[test_item_ids, :]
# 		if sp.issparse(self.S_tr):
# 			self.S_tr = self.S_tr.todense()
# 			self.S_te = self.S_te.todense()
#
# 		self.X_tr = self.R_train.todense().T
# 		self.U_tr = self.R_train.todense()
#
# 		self.X_te = self.R_test.todense().T
# 		self.U_te = self.R_test.todense()
#
# 		self.n_users = self.X_tr.shape[0] + self.X_te.shape[0]
# 		self.n_items = self.U_tr.shape[0]
#
#
# 	def item_raw_load(self, path):
# 		item_raw_data = pd.read_csv(path, delimiter=",", header=0, encoding='utf-8')
# 		# item_id = item_raw_data['item_id']
# 		content_data = item_raw_data['title'] + ' ' + item_raw_data['description'] + ' ' + item_raw_data['feature']
# 		# content_text = content_data.tolist()
# 		from sklearn.feature_extraction.text import TfidfVectorizer
# 		vectorizer = TfidfVectorizer(min_df=10, max_df=0.1, stop_words='english', max_features=10000)
# 		text_tfidf = vectorizer.fit_transform(content_data.apply(lambda x: np.str_(x)))
# 		return text_tfidf
#
# 	def record_list(self, train_u, train_i):
#
# 		train_items = {}
# 		for i in range(len(train_u)):
# 			u = train_u[i]
# 			if u not in train_items.keys():
# 				train_items[u] = [train_i[i]]
# 			else:
# 				train_items[u].append(train_i[i])
#
# 		return train_items
#
# 	def get_batch(self, batch_ids, content_embedding):
#
# 		batch_user = list(np.array(self.train_u)[batch_ids])
# 		batch_item = list(np.array(self.train_i)[batch_ids])
# 		batch_X_tr = self.X_tr[batch_item, :]
# 		batch_U_tr = self.U_tr
# 		batch_S_tr_tfidf = self.S_tr[batch_item, :]
# 		# batch_S_tr = self.S_tr[batch_item, :]
# 		batch_S_tr_bert = content_embedding[batch_item, :]
# 		batch_S_tr = [batch_S_tr_tfidf, batch_S_tr_bert]
# 		return batch_user, batch_item, batch_X_tr, batch_U_tr, batch_S_tr
# 		# return batch_user, batch_item, batch_X_tr, batch_U_tr, batch_bert_tr_inputs
#
#
# 	def sample_iterator(self, batch_ids):
#
#
# 		batch_user = list(np.array(self.train_u)[batch_ids])
# 		batch_item = list(np.array(self.train_i)[batch_ids])
# 		pos_user = batch_user
# 		neg_user = []
# 		for i in batch_item:
# 			# pos_items += self.sample_pos_items(u, 1)
# 			neg_user += self.sample_neg_items(i, 1)
#
# 		batch_X_tr = self.X_tr[batch_item, :]
# 		batch_U_tr = self.U_tr
# 		batch_S_tr = self.S_tr[batch_item, :]
#
# 		return batch_item, pos_user, neg_user, batch_X_tr, batch_U_tr, batch_S_tr
#
#
#
# 	def sample_neg_items(self, u, num):
# 		neg_items = list(set(range(self.R_train.shape[0])) - set(self.train_items[u]))
# 		return rd.sample(neg_items, num)

class Data_CiteU(object):
	def __init__(self, path, batch_size):

		self.batch_size = batch_size

		split_folder = os.path.join(path, 'cold')

		item_content_file = os.path.join(path, 'item_features_0based.txt')
		train_file = os.path.join(split_folder, 'train.csv')
		test_cold_file = os.path.join(split_folder, 'test.csv')
		test_cold_iid_file = os.path.join(split_folder, 'test_item_ids.csv')

		# load split
		# timer.tic()
		train = pd.read_csv(train_file, delimiter=",", header=None, dtype=np.int32).values.ravel().view(
			dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32)])
		train_user_ids = np.unique(train['uid'])

		train_item_ids = np.unique(train['iid'])
		self.train_item_ids = train_item_ids

		train_user_ids_map = {user_id: i for i, user_id in enumerate(train_user_ids)}
		train_item_ids_map = {user_id: i for i, user_id in enumerate(train_item_ids)}

		_train_i_for_inf = [train_user_ids_map[_t[0]] for _t in train]
		_train_j_for_inf = [train_item_ids_map[_t[1]] for _t in train]
		self.R_train = sp.coo_matrix(
			(np.ones(len(_train_i_for_inf)),
			 (_train_i_for_inf, _train_j_for_inf)),
			shape=[len(train_user_ids), len(train_item_ids)]).tolil(copy=False)

		self.train_items = self.record_list(_train_j_for_inf, _train_i_for_inf)
		self.train_u = _train_i_for_inf
		self.train_i = _train_j_for_inf

		with open(test_cold_iid_file) as f:
			test_item_ids = [int(line) for line in f]
			self.test_item_ids = test_item_ids
			test_data = pd.read_csv(test_cold_file, delimiter=",", header=None, dtype=np.int32).values.ravel()
			test_data = test_data.view(
				dtype=[('uid', np.int32), ('iid', np.int32), ('inter', np.int32)])

			test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}

			_test_ij_for_inf = [(t[0], t[1]) for t in test_data if t[1] in test_item_ids_map]
			# test_user_ids
			test_user_ids = np.unique(test_data['uid'])
			# test_user_ids_map
			# test_user_ids_map = {user_id: i for i, user_id in enumerate(test_user_ids)}
			test_user_ids_map = train_user_ids_map

			_test_i_for_inf = [test_user_ids_map[_t[0]] for _t in _test_ij_for_inf]
			_test_j_for_inf = [test_item_ids_map[_t[1]] for _t in _test_ij_for_inf]
			self.R_test = sp.coo_matrix(
				(np.ones(len(_test_i_for_inf)),
				 (_test_i_for_inf, _test_j_for_inf)),
				shape=[len(train_user_ids), len(test_item_ids)]).tolil(copy=False)

			self.test_items = self.record_list(_test_j_for_inf, _test_i_for_inf)
			self.test_u = _test_i_for_inf
			self.test_users = self.record_list(_test_i_for_inf, _test_j_for_inf)


		item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)

		item_content = tfidf(item_content)

		from sklearn.utils.extmath import randomized_svd
		u, s, _ = randomized_svd(item_content, n_components=300, n_iter=5)
		item_content = u * s
		_, item_content = utils.prep_standardize(item_content)

		if sp.issparse(item_content):
			item_feature = item_content.tolil(copy=False)
		else:
			item_feature = item_content
		# timer.toc('loaded item feature sparse matrix: %s' % (str(item_content.shape))).tic()
		self.item_content = item_feature


		self.S_tr = item_feature[train_item_ids, :]
		self.S_te = item_feature[test_item_ids, :]
		if sp.issparse(self.S_tr):
			self.S_tr = self.S_tr.todense()
			self.S_te = self.S_te.todense()



		self.X_tr = self.R_train.todense().T
		self.U_tr = self.R_train.todense()

		self.X_te = self.R_test.todense().T
		self.U_te = self.R_test.todense()


		self.n_users = self.X_tr.shape[0] + self.X_te.shape[0]
		self.n_items = self.U_tr.shape[0]



	def record_list(self, train_u, train_i):
		# nonzeros = np.nonzero(X)
		# train_u = list(nonzeros[0])
		# train_i = list(nonzeros[1])
		train_items = {}
		for i in range(len(train_u)):
			u = train_u[i]
			if u not in train_items.keys():
				train_items[u] = [train_i[i]]
			else:
				train_items[u].append(train_i[i])

		return train_items

	def get_batch(self, batch_ids):

		batch_user = list(np.array(self.train_u)[batch_ids])
		batch_item = list(np.array(self.train_i)[batch_ids])
		batch_X_tr = self.X_tr[batch_item, :]
		batch_U_tr = self.U_tr
		batch_S_tr = self.S_tr[batch_item, :]

		return batch_user, batch_item, batch_X_tr, batch_U_tr, batch_S_tr






def tfidf(R):
	row = R.shape[0]
	col = R.shape[1]
	Rbin = R.copy()
	Rbin[Rbin != 0] = 1.0
	R = R + Rbin
	tf = R.copy()
	tf.data = np.log(tf.data)
	idf = np.sum(Rbin, 0)
	idf = np.log(row / (1 + idf))
	idf = sp.spdiags(idf, 0, col, col)
	return tf * idf
