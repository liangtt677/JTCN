import numpy as np
import tensorflow as tf
import os
import re
from tqdm import tqdm
# from tensorflow.python.keras import backend as K



def squash(inputs, axis=2, ord="euclidean", name=None):
    name = "squashing" if name is None else name
    with tf.name_scope(name):
        norm = tf.norm(inputs, ord=ord, axis=axis, keepdims=True, name=name)
        norm_squared = tf.square(norm)
        scalar_factor = norm_squared / (1 + norm_squared)
        return scalar_factor * (inputs / norm)


class CSN(object):
    def __init__(self, args, data_config):
        self.user_num = data_config['n_users']
        self.item_num = data_config['n_items']
        self.uin_dim = data_config['uin_dim']
        self.iin_dim = data_config['iin_dim']
        self.feat_dim = data_config['feat_dim']
        self.seed = data_config['seed']

        self.emb_dim = args.embed_size
        # self.ac_emb_dim = 2 * args.embed_size
        # self.z_dim = args.embed_size
        self.lr = args.lr
        self.regs = eval(args.regs)[0]
        self.batch_size = args.batch_size
        self.attention_dim = 128

        self.CapNum = 3
        self._build_inputs()
        self._build_model()
        

    def _build_inputs(self):
        with tf.name_scope('input'):
            self.user_history = tf.placeholder(
                dtype=tf.float32, shape=[None, self.uin_dim], name='user_history')
            if self.iin_dim != None:
                self.item_history = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.iin_dim], name='item_history')
            self.user_social = tf.placeholder(
        		dtype=tf.float32, shape=(None, self.feat_dim), name='user_social')

            self.pos_items = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='item_ids')
            self.neg_items = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='neg_item_ids')



    def _init_weights(self):
        all_weights = dict()
        with tf.name_scope('embedding'):
            wlimit = np.sqrt(6.0 / (2 * self.emb_dim + self.emb_dim))
            initializer = tf.contrib.layers.xavier_initializer(seed=self.seed)
            # initializer = tf.random_normal_initializer(mean=0., stddev=0.01)
            # initializer = tf.random_uniform_initializer(-wlimit, wlimit)

            # all_weights['user_embedding'] = tf.Variable(initializer([self.user_num, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.item_num, self.emb_dim]), name='item_embedding_csr')
            # all_weights['user_embedding_extra'] = tf.Variable(initializer([self.user_num, self.emb_dim]), name='user_embedding_extra')

            all_weights['fc_w'] = tf.Variable(tf.random_uniform([2*self.emb_dim, self.emb_dim], -wlimit, wlimit, seed=self.seed), name='fc_w_csr')
            all_weights['fc_b'] = tf.Variable(tf.random_uniform([self.emb_dim], -wlimit, wlimit, seed=self.seed), name='fc_b_csr')

            # # attention params
            np.random.seed(self.seed)
            glorot = np.sqrt(2.0 / (self.emb_dim + self.emb_dim))
            all_weights['attention_W'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.emb_dim, self.attention_dim)),
                dtype=np.float32, name="attention_W")  # K * AK
            all_weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(1, self.attention_dim)), dtype=np.float32,
                name="attention_b")  # 1 * AK
            all_weights['attention_p'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.attention_dim)), dtype=np.float32,
                name="attention_p")  # AK

        with tf.name_scope('Routing'):
            # all_weights['S'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
            all_weights['S'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.emb_dim, self.emb_dim)), dtype=np.float32)
            # all_weights['S'] = tf.Variable(tf.random.normal(mean=0, stddev=1, shape=[self.emb_dim, self.emb_dim], seed=self.seed),
            #                                dtype=np.float32)


        print('using xavier initialization')

        return all_weights

    def _build_model(self):
        self.weights = self._init_weights()

        regularizer = tf.contrib.layers.l2_regularizer(self.regs)


        self.concat_feature = self.user_social
        self.user_feature_embed = tf.layers.dense(self.concat_feature, self.emb_dim, kernel_regularizer=regularizer,
                                                  bias_regularizer=regularizer, activation=tf.nn.relu, use_bias=True)
        self.user_featureExtra_embed = tf.layers.dense(self.concat_feature, self.emb_dim, kernel_regularizer=regularizer,
                                                       bias_regularizer=regularizer, activation=tf.nn.relu,
                                                       use_bias=True)


        if self.iin_dim != None:
            self.user_pref_embed = tf.layers.dense(self.user_history, self.emb_dim, kernel_regularizer=regularizer, bias_regularizer=regularizer, activation=tf.nn.relu, use_bias=True)
            self.item_pref_embed = tf.layers.dense(self.item_history, self.emb_dim, kernel_regularizer=regularizer, bias_regularizer=regularizer, activation=tf.nn.relu, use_bias=True)

            self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_pref_embed, self.pos_items)

        else:
            user_history_extend = tf.tile(tf.expand_dims(self.user_history, 2), [1, 1, self.emb_dim])
            item_embed_extend = tf.tile(tf.expand_dims(self.weights['item_embedding'], 0), [self.batch_size, 1, 1])
            user_history_embed = tf.multiply(user_history_extend, item_embed_extend)
            B = tf.random_normal_initializer(mean=0., stddev=1, seed=self.seed)([self.batch_size, self.item_num, self.CapNum])
            for iter in range(3):
                W = tf.nn.softmax(B, axis=2)
                # S_e = tf.tile(tf.expand_dims(tf.matmul(user_history_embed, S), -1), [1, 1, 1, CapNum])
                # W_e = tf.tile(tf.expand_dims(W, 2), [1, 1, self.emb_dim, 1])
                S_e = tf.transpose(tf.matmul(user_history_embed, self.weights['S']), perm=[0, 2, 1])
                interest_cap = tf.transpose(tf.matmul(S_e, W), perm=[0, 2, 1])
                # self.user_history_embed = tf.transpose(tf.reduce_sum(tf.multiply(W_e, S_e), 1), perm=[0, 2, 1])
                interest_cap_squash = squash(interest_cap)
                update_b = tf.transpose(tf.matmul(interest_cap_squash, S_e), perm=[0, 2, 1])
                B = B + update_b

            self.user_pref_embed = self._attention_layer(interest_cap_squash)

        self.user_concat_embed = tf.concat([self.user_feature_embed, self.user_pref_embed], axis=1)
        # self.user_concat_embed = tf.concat([self.cross_feature, self.user_pref_embed], axis=1)
        self.user_concat_embed = tf.nn.dropout(self.user_concat_embed, 0.5)
        # self.user_concat_embed = self._cross_gate(self.user_feature_embed, self.user_history_mean)
        self.user_final = tf.nn.relu(tf.matmul(self.user_concat_embed, self.weights['fc_w']) + self.weights['fc_b'])
        # self.user_final = tf.nn.relu(tf.matmul(self.user_final, self.weights['fc_w2']) + self.weights['fc_b2'])


        if self.iin_dim != None:
            self.pred = tf.matmul(self.user_final, self.item_pref_embed, transpose_b=True)
        else:
            self.pred = tf.matmul(self.user_final, self.weights['item_embedding'], transpose_b=True)



        self.pos_item_onehot = tf.one_hot(self.pos_items, depth=self.item_num, axis=1)


        self._loss()

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.batch_ratings = self._test_rating()


    def _attention_layer(self, embedding):
        # user_history_extend = tf.tile(tf.expand_dims(user_history, 2), [1, 1, self.emb_dim])
        # item_embed_extend = tf.tile(tf.expand_dims(self.weights['item_embedding'], 0), [self.batch_size, 1, 1])
        # user_history_embed = tf.multiply(user_history_extend, item_embed_extend)

        attention_mul = tf.reshape(tf.matmul(tf.reshape(embedding, shape=[-1, self.emb_dim]), \
                        self.weights['attention_W']), shape=[-1, self.CapNum, self.attention_dim])
        attention_exp = tf.exp(tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(attention_mul + \
                        self.weights['attention_b'])), 2, keep_dims=True))  # None * (M'*(M'-1)) * 1
        attention_sum = tf.reduce_sum(attention_exp, 1, keep_dims=True)  # None * 1 * 1
        attention_out = tf.div(attention_exp, attention_sum, name="attention_out")  # None * (M'*(M'-1)) * 1
        # attention_out = tf.nn.dropout(attention_out, 0.8)  # dropout

        output_sum = tf.reduce_sum(tf.multiply(attention_out, embedding), 1, name="afm")  # None * K
        # output_sum = tf.nn.dropout(output_sum, 0.8)  # dropout

        return output_sum

    # def _self_attention_layer(self, embedding):
    #     # user_history_extend = tf.tile(tf.expand_dims(user_history, 2), [1,1,self.emb_dim])
    #     # item_embed_extend = tf.tile(tf.expand_dims(self.weights['item_embedding'],0), [self.batch_size, 1, 1])
    #     # user_history_embed = tf.multiply(user_history_extend, item_embed_extend)
    #     #
    #     # # Q = tf.matmul(user_history_embed, self.weights['a_fc_w1'])
    #     # # K = tf.matmul(user_history_embed, self.weights['a_fc_w2'])
    #
    #     Q = tf.layers.dense(embedding, self.emb_dim, use_bias=True)  # (N, T_q, d_model)
    #     K = tf.layers.dense(embedding, self.emb_dim, use_bias=True)  # (N, T_k, d_model)
    #
    #     outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    #     outputs /= self.emb_dim ** 0.5
    #
    #     # mask diagonal elements
    #     # ones = tf.ones((self.item_num, self.item_num)) - tf.eye(self.item_num)
    #     ones = tf.ones((self.CapNum, self.CapNum)) - tf.eye(self.CapNum)
    #     ones = tf.tile(tf.expand_dims(ones, 0), [self.batch_size, 1, 1])
    #     outputs = tf.multiply(ones, outputs)
    #
    #     outputs = tf.nn.softmax(outputs)
    #     outputs = tf.layers.dropout(outputs, rate=0.5, training=True)
    #     outputs = tf.matmul(outputs, embedding)
    #
    #     output_sum = tf.reduce_sum(outputs, 1)
    #
    #     return output_sum


    def _loss(self):
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.pos_item_onehot, logits=self.pred)
        self.base_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.pos_item_onehot, logits=self.pred))


        diff = tf.subtract(self.user_pref_embed, self.user_featureExtra_embed)
        # diff2 = tf.subtract(self.interest_cap, self.user_feature_caps)
        self.diff_loss = tf.nn.l2_loss(diff) / self.batch_size
        # self.diff_loss = self.diff_loss + tf.nn.l2_loss(diff2) / self.batch_size

        reg_lam = self.regs
        self.regularizer = tf.contrib.layers.l2_regularizer(reg_lam)(self.weights['fc_w']) + \
                           tf.contrib.layers.l2_regularizer(reg_lam)(self.weights['fc_b']) + \
                           tf.contrib.layers.l2_regularizer(reg_lam)(self.weights['item_embedding'])

        self.loss = self.base_loss + self.diff_loss + self.regularizer


    def _test_rating(self):
        concat_user = tf.concat([self.user_feature_embed, self.user_featureExtra_embed], axis=1)
        user_embed_cold = tf.nn.relu(tf.matmul(concat_user, self.weights['fc_w']) + self.weights['fc_b'])

        if self.iin_dim != None:
            test_pred = tf.matmul(user_embed_cold, self.item_pref_embed, transpose_b=True)
        else:

            test_pred = tf.matmul(user_embed_cold, self.weights['item_embedding'], transpose_b=True)

        return test_pred


