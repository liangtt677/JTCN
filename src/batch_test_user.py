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
    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
# N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


# BATCH_SIZE = len(data_generator.train_u)
# np.savetxt("test_content.txt", data_generator.S_te)

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


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.test_items[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    # item_score = []
    # for i in test_items:
    #     item_score.append((i, rating[i]))

    # item_score = sorted(item_score, key=lambda x: x[1])
    # item_score.reverse()
    # item_sort = [x[0] for x in item_score]

    # r = []
    # for i in item_sort:
    #     if i in user_pos_test:
    #         r.append(1)
    #     else:
    #         r.append(0)

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    # auc = 0
    return get_performance(user_pos_test, r, auc, Ks)


def test_one_user_cold(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]

    # user u's items in the test set
    user_pos_test = data_generator.test_users[u]

    all_items = set(range(data_generator.S_te.shape[0]))
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
    user_pos_test = data_generator.test_users[u]

    all_items = set(range(data_generator.S_te.shape[0]))
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


def test_pop(users_to_test, item_rate):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    test_users = users_to_test
    n_test_users = len(test_users)

    user_batch_rating_uid = zip(item_rate, test_users)
    batch_result = pool.map(test_one_user, user_batch_rating_uid)

    for re in batch_result:
        result['precision'] += re['precision'] / n_test_users
        result['recall'] += re['recall'] / n_test_users
        result['ndcg'] += re['ndcg'] / n_test_users
        result['hit_ratio'] += re['hit_ratio'] / n_test_users
        result['auc'] += re['auc'] / n_test_users

    pool.close()
    return result


def test(sess, model, items_to_test, content_embedding, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    # u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_items = items_to_test
    n_test_items = len(test_items)
    # u_batch_size = n_test_users
    n_item_batchs = n_test_items // i_batch_size + 1

    count = 0
    rating_list = []
    for u_batch_id in tqdm(range(n_item_batchs)):
        start = u_batch_id * i_batch_size
        end = (u_batch_id + 1) * i_batch_size

        item_batch = test_items[start: end]
        # print(user_batch)

        # if batch_test_flag:

        #     n_item_batchs = ITEM_NUM // i_batch_size + 1
        #     rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

        #     i_count = 0
        #     for i_batch_id in range(n_item_batchs):
        #         i_start = i_batch_id * i_batch_size
        #         i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

        #         item_batch = range(i_start, i_end)

        #         if drop_flag == False:
        #             i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
        #                                                         model.pos_items: item_batch})
        #         else:
        #             i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
        #                                                         model.pos_items: item_batch,
        #                                                         model.node_dropout: [0.]*len(eval(args.layer_size)),
        #                                                         model.mess_dropout: [0.]*len(eval(args.layer_size))})
        #         rate_batch[:, i_start: i_end] = i_rate_batch
        #         i_count += i_rate_batch.shape[1]

        #     assert i_count == ITEM_NUM

        # else:
        # item_batch = range(ITEM_NUM)

        batch_S_te = data_generator.S_te[item_batch]
        # print(batch_S_te.shape)

        if args.dataset == 'CiteU':
            # batch_bert_te_inputs = [inputs[item_batch, :] for inputs in data_generator.bert_test_inputs]
            batch_S_te_bert = content_embedding[item_batch, :]
            # batch_S_te = np.hstack((data_generator.S_te[user_batch], content_embedding[user_batch, :]))
            feed_dict = {
                model.user_social: batch_S_te,
                model.user_social_bert: batch_S_te_bert
                # model.in_id: batch_bert_te_inputs[0],
                # model.in_mask: batch_bert_te_inputs[1],
                # model.in_segment: batch_bert_te_inputs[2]
            }

            rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)

            # test_u_no = list(set(range(ITEM_NUM)) - set(data_generator.test_u))
            # rate_batch[:, test_u_no] = 0

            rating_list.append(rate_batch)
            # user_batch_rating_uid = zip(rate_batch, user_batch)
            # batch_result = pool.map(test_one_user_cold_citeu, user_batch_rating_uid)
        else:
            rate_batch = sess.run(model.batch_ratings, feed_dict={model.user_social: batch_S_te})
            # print(rate_batch)
            user_batch_rating_uid = zip(rate_batch, user_batch)
            batch_result = pool.map(test_one_user_cold, user_batch_rating_uid)

    # count += len(batch_result)
    # print('########')
    # print(len(batch_result))
    user_batch_rating = np.vstack(rating_list).transpose()
    # print(user_batch_rating.shape)
    user_set = list(data_generator.test_users.keys())
    # print(user_set)
    user_batch_rating = user_batch_rating[user_set, :]
    print(user_batch_rating.shape)
    user_batch_rating_uid = zip(user_batch_rating, np.array(user_set))
    batch_result = pool.map(test_one_user_cold_citeu, user_batch_rating_uid)

    K_max = []
    for re in batch_result:
        result['precision'] += re['precision'] / len(user_set)
        result['recall'] += re['recall'] / len(user_set)
        result['ndcg'] += re['ndcg'] / len(user_set)
        result['hit_ratio'] += re['hit_ratio'] / len(user_set)
        result['auc'] += re['auc'] / len(user_set)
        K_max.append(re['K_max'])

    # assert count == n_test_users
    pool.close()
    return result, K_max
