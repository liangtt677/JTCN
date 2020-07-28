import tensorflow as tf
import numpy as np
import sys
import os.path
import os
from tqdm import tqdm
import datetime
import scipy.io as io

from helper import *
from batch_test import *
import utils
from csn import CSN

def train(args):
    seed = 2333
    tf.set_random_seed(seed)

    config = dict()
    config['seed'] = seed
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['uin_dim'] = data_generator.X_tr.shape[1]
    config['feat_dim'] = data_generator.S_tr.shape[1]
    config['iin_dim'] = None



    t0 = time()

    configp = tf.ConfigProto()
    configp.gpu_options.allow_growth = True
    sess = tf.Session(config=configp)

    # K.set_session(sess) % delete
    # # sess = tf.Session()
    # # set_session(sess)

    model = CSN(args=args, data_config=config)

    """
    *********************************************************
    Save the model parameters.
    """
    save_saver = tf.train.Saver()

    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/l%s_r%s' % (args.weights_path, args.dataset,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)



    # uninitialized_vars = []
    # for var in tf.global_variables():
    #     try:
    #         sess.run(var)
    #         # print(var)
    #     except tf.errors.FailedPreconditionError:
    #         uninitialized_vars.append(var)
    #
    # # print(uninitialized_vars)
    # # for var in uninitialized_vars:
    #     # print(var)
    #
    # print('######initialize######')
    # init = tf.variables_initializer(uninitialized_vars)
    # sess.run(init)

    sess.run(tf.global_variables_initializer())

    cur_best_pre_0 = 0.


    """
    *********************************************************
    Train.
    """

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    total_batch = int(len(data_generator.train_u) / BATCH_SIZE)

    save_path = '%soutput/test/%s_l%s_r%s_dim%d_bs%d.result' % (
    args.proj_path, args.dataset, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), args.embed_size, BATCH_SIZE)

    ensureDir(save_path)

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, diff_loss, dc_loss, generator_loss = 0., 0., 0., 0., 0.
        rand_idx = np.arange(len(data_generator.train_u))
        np.random.shuffle(rand_idx)
        rand_idx = list(rand_idx)
        # train_d = False
        for i in tqdm(range(total_batch)):
            batch_ids = rand_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            batch_user, batch_item, batch_X_tr, batch_U_tr, batch_S_tr = data_generator.get_batch(batch_ids)
            feed_dict = {model.user_history: batch_X_tr,
                         model.user_social: batch_S_tr,
                         model.pos_items: batch_user
                         }

            _, batch_loss, batch_base_loss, batch_diff_loss, pred = sess.run([model.opt, model.loss, model.base_loss, model.diff_loss, model.pred],
                                                                feed_dict=feed_dict)

            loss = batch_loss
            base_loss = batch_base_loss
            diff_loss = batch_diff_loss



        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 1 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, base_loss, diff_loss)
                print(perf_str)
            continue

        t2 = time()
        # print('############Test Start############')
        users_to_test = list(data_generator.test_items.keys())
        ret = test(sess, model, users_to_test)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, base_loss, diff_loss,
                        ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3], ret['recall'][4],
                        ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3], ret['precision'][4],
                        ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4])
            print(perf_str)

            f = open(save_path, 'a')
            f.write(perf_str+'\n')
            f.close()

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=10)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)


    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)


    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, embed_size=%s, regs=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.embed_size, args.regs, final_perf))
    f.close()




if __name__ == '__main__':

    train(args)
