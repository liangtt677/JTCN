import metrics as metrics
from parser import parse_args
from data_loader import *
import multiprocessing
import heapq
from tqdm import tqdm

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
if args.dataset == 'CiteU':
    data_generator = Data_CiteU(path=args.data_path + args.dataset, batch_size=args.batch_size)
else:
    print('Amazon')
    data_generator = Data_Amazon(path=args.data_path + args.dataset, batch_size=args.batch_size)

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc, K_max_item_score

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc, K_max_item_score

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        # hit_ratio.append(metrics.hit_at_k(r, K))
        hit_ratio.append(metrics.average_precision(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}



def test_one_user_cold(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    
    #user u's items in the test set
    user_pos_test = data_generator.test_items[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items)

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    # auc = 0
    return get_performance(user_pos_test, r, auc, Ks)


def test_one_user_cold_citeu(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]

    # user u's items in the test set
    user_pos_test = data_generator.test_items[u]

    all_items = set(range(data_generator.U_te.shape[0]))

    test_items = list(all_items)

    if args.test_flag == 'part':
        r, auc, K_max_item = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc, K_max_item = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    # auc = 0
    result = get_performance(user_pos_test, r, auc, Ks)
    result['K_max'] = K_max_item

    # return get_performance(user_pos_test, r, auc, Ks)
    return result
# def test_pop(users_to_test, item_rate):
#     result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
#               'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
#
#     pool = multiprocessing.Pool(cores)
#
#     test_users = users_to_test
#     n_test_users = len(test_users)
#
#     user_batch_rating_uid = zip(item_rate, test_users)
#     batch_result = pool.map(test_one_user, user_batch_rating_uid)
#
#
#     for re in batch_result:
#         result['precision'] += re['precision']/n_test_users
#         result['recall'] += re['recall']/n_test_users
#         result['ndcg'] += re['ndcg']/n_test_users
#         result['hit_ratio'] += re['hit_ratio']/n_test_users
#         result['auc'] += re['auc']/n_test_users
#
#
#     pool.close()
#     return result


def test(sess, model, users_to_test):
    # result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
    #           'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    # u_batch_size = n_test_users
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        batch_S_te = data_generator.S_te[user_batch]

        feed_dict = {model.user_social: batch_S_te
        }

        rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)

        test_u_no = list(set(range(ITEM_NUM)) - set(data_generator.test_u))
        rate_batch[:, test_u_no] = 0
        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user_cold_citeu, user_batch_rating_uid)

        # if args.dataset == 'CiteU':
        #     # rate_batch = sess.run(model.batch_ratings, feed_dict={model.user_social: batch_S_te, model.item_history: data_generator.U_tr})
        #     # rate_batch = sess.run(model.batch_ratings,
        #     #                       feed_dict={model.user_social: batch_S_te})
        #
        #     batch_bert_te_inputs = [inputs[user_batch, :] for inputs in data_generator.bert_test_inputs]
        #     batch_S_te_bert = content_embedding[user_batch, :]
        #     # batch_S_te = np.hstack((data_generator.S_te[user_batch], content_embedding[user_batch, :]))
        #     feed_dict = {
        #                   model.user_social: batch_S_te,
        #                   model.user_social_bert: batch_S_te_bert,
        #                   model.in_id: batch_bert_te_inputs[0],
        #                   model.in_mask: batch_bert_te_inputs[1],
        #                   model.in_segment: batch_bert_te_inputs[2]
        #                  }
        #
        #     rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)
        #
        #     test_u_no = list(set(range(ITEM_NUM)) - set(data_generator.test_u))
        #     rate_batch[:, test_u_no] = 0
        #     user_batch_rating_uid = zip(rate_batch, user_batch)
        #     batch_result = pool.map(test_one_user_cold_citeu, user_batch_rating_uid)
        # elif args.dataset == 'Amazon':
        #     batch_S_te_bert = content_embedding[user_batch, :]
        #     feed_dict = {
        #         model.user_social: batch_S_te,
        #         model.user_social_bert: batch_S_te_bert
        #     }
        #
        #     rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)
        #
        #     test_u_no = list(set(range(ITEM_NUM)) - set(data_generator.test_u))
        #     rate_batch[:, test_u_no] = 0
        #     user_batch_rating_uid = zip(rate_batch, user_batch)
        #     batch_result = pool.map(test_one_user_cold_citeu, user_batch_rating_uid)
        #
        # else:
        #     rate_batch = sess.run(model.batch_ratings, feed_dict={model.user_social: batch_S_te})
        #
        #     user_batch_rating_uid = zip(rate_batch, user_batch)
        #     batch_result = pool.map(test_one_user_cold, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users



    assert count == n_test_users
    pool.close()
    return result
