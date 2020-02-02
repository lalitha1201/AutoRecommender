import pandas as pd
import data_preprocessing as ds
import metrics
from sklearn.model_selection import KFold
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import model_selection
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
import get_top as top

import recommendations_cf as rec
import matrix_factorization_gridsearch as mf


def select_model(loaded_data, model_selection='user_user'):
    # default model is user-user based collaborative filtering
    if model_selection == 'user_user':
        algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
    elif model_selection == 'item_item':
        algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
    else:
        algo = mf.matrix_factorization_param(loaded_data)

    return algo
import os, io
 
def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
 
    file_name = (os.path.expanduser('~') +
                 'sample_us.tsv')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
 
    return rid_to_name



def all_models(_file_path,modelname):
    #data = ds.get_data(_file_path, 'data/data_subset.csv', 0.99)
    data = ds.get_data(_file_path)
    
    data_surprise = data[['customer_id', 'product_id', 'star_rating']]. \
    rename(columns={'customer_id': 'userID', 'product_id': 'itemID', 'star_rating': 'rating'})

    reader = Reader(rating_scale=(1.0, 5.0))
    df_loaded = Dataset.load_from_df(data_surprise, reader)
    #trainset = df_loaded.build_full_trainset()

    results_list = []


    # features
    reviews = data.shape[0]
    n_users = data.customer_id.nunique()
    n_products = data.product_id.nunique()
    mean_rating = data.star_rating.mean()
    rating_std = data.star_rating.std()
    sparsity = reviews * 100 / (n_users * n_products)

    for model in ['user_user', 'item_item', 'matrix_fact']:
        # Perform cross validation
        results = model_selection.cross_validate(select_model(df_loaded, model_selection=modelname),
                                                 df_loaded,
                                                 measures=['RMSE', 'MAE'], cv=5, verbose=False)

        # precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
        kf = KFold(n_splits=5)
        trainset, testset = train_test_split(df_loaded, test_size=.25)
        map_k, mar_k = 0, 0
        algo = select_model(df_loaded, model_selection=model)
        #for trainset, testset in trainset.split():
        algo.fit(trainset)
        predictions = algo.test(testset)
        top_n = rec.get_top_n(predictions, n=30)
        top_n
       # top_n = rec.get_top_n(predictions,data_surprise,userID = 11613707)
       # pred_SVD_124 = top.get_top_n(predictions,userId = 13545982,data = data)
        #top_n.head(15)
       # pred_SVD_124
        print('Recommendations for the user')
        for uid, user_ratings in top_n.items():
           # if uid == 43173394:
            # print('Recommendations for the user')
             print(uid, [iid for (iid, _) in user_ratings])
             precisions, recalls = metrics.precision_recall_at_k(predictions, k=5, threshold=4)

            # Precision and recall can then be added for all the splits

        map_k += precisions
        mar_k += recalls
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series(map_k / 5, index=['map_k']))
        tmp = tmp.append(pd.Series(mar_k / 5, index=['mar_k']))
        tmp = tmp.append(pd.Series([str(_file_path)], index=['data']))
        tmp = tmp.append(pd.Series([str(model)], index=['Algorithm']))

        # features
        tmp = tmp.append(pd.Series(reviews, index=['reviews']))
        tmp = tmp.append(pd.Series(n_users, index=['n_users']))
        tmp = tmp.append(pd.Series(n_products, index=['n_products']))
        tmp = tmp.append(pd.Series(mean_rating, index=['mean_rating']))
        tmp = tmp.append(pd.Series(rating_std, index=['std_rating']))
        tmp = tmp.append(pd.Series(sparsity, index=['sparsity']))

        results_list.append(tmp)
        # print(results_list)
    results_df = pd.DataFrame(results_list)

    # saving the results file to folder
    return results_df
