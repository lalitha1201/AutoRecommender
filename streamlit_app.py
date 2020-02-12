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
import streamlit as st
import recommendations_cf as rec
import matrix_factorization_gridsearch as mf
def customer_recomendation(UserId):
    if UserId not in merge1.index:
        print('Customer not found.')
        return UserId
    return merge1.loc[UserId]

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
 


st.title('Welcome to RecServe!')
st.header('Let me help you with the product recommendations')
option1 = st.selectbox(
    'Select the path for the dataset?',
    ['sample_us.tsv'])
#st.write('sample_us.tsv')
#st.write('You selected:',option1)

#url = st.text_input('Enter the path for the data')
st.write('The data is loaded')
#data_load_state = st.text('Loading the data')
data = ds.get_data(option1)
#st.write(data)

    #data = ds.get_data(_file_path, 'data/data_subset.csv', 0.99)
#data = ds.get_data('/Users/lalitharahul/Desktop/AutoRecommender/RecServe/sample_us.tsv')
#data = ds.get_data(url)
#st.write(data)
#data_load_state.text('Data is preprocessed')
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
url = st.sidebar.text_input('Enter CustomerID')
#if st.button('Enter'):
   # st.write('Entered customer id')
#st.write('The Entered Customer Id is', url)
url1 = st.sidebar.text_input('Enter ProductId')
#st.write('The Entered Product Id is', url1)
if url in ['18778586','24769659','44331596'] and url1 in ['B00EDBY7X8','B00D7JFOPC','B002LHA74O']:
   option = st.sidebar.selectbox(
       'Select the following?',
        ['Recommend items for users to purchase?','Recommend similar items for users to purchase?'])
#st.write('You selected:', option)
  
   if option == 'Recommend items for users to purchase?':
      option1 = 'user_user'
      st.write('Best Algorithm: KNNWithMeans')
   else:
      option1 = 'item_item'
      st.write('Best Algorithm:SlopeOne')
#for model in ['user_user', 'item_item', 'matrix_fact']:
        # Perform cross validation
   results = model_selection.cross_validate(select_model(df_loaded, model_selection=option1),
                                                 df_loaded,
                                                 measures=['RMSE', 'MAE'], cv=5, verbose=False)

        # precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)
   kf = KFold(n_splits=5)
   trainset, testset = train_test_split(df_loaded, test_size=.25)
   map_k, mar_k = 0, 0
   algo = select_model(df_loaded, model_selection=option)
        #for trainset, testset in trainset.split():
   algo.fit(trainset)
   predictions = algo.test(testset)
   st.header('Recommendations for the users:')
    #st.write(predictions)
        #uid = str(11613707)  # raw user id (as in the ratings file). They are **strings**!
#iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
        #pred = algo.predict(uid,verbose=True)
        #pred
   top_n = rec.get_top_n(predictions, n=3)
        
        #top_n[data_surprise.userID[38745832]]
 
       # top_n = rec.get_top_n(predictions,data_surprise,userID = 11613707)
       # pred_SVD_124 = top.get_top_n(predictions,userId = 13545982,data = data)
        #top_n.head(15)
       # pred_SVD_124
    #st.write(top_n)
   print('Recommendations for the user')
   dfo = pd.DataFrame(columns=['UserId', 'ItemId'])
   i=0
        
   for uid, user_ratings in top_n.items():
           
            # print('Recommendations for the user')
       # st.write('user ID,Item Id')
       row = [uid, top_n[uid]]
       dfo.loc[i] = row
       i=i+1
        #st.write(uid, [iid for (iid, _) in user_ratings])
       print('user Id ,Item Id')
       print(uid, [iid for (iid, _) in user_ratings])

       precisions, recalls = metrics.precision_recall_at_k(predictions, k=5, threshold=4)

            # Precision and recall can then be added for all the splits
   merge = dfo.merge(data,left_on='UserId',right_on= 'customer_id')
   merge1 = merge[['product_title']]
   st.write(merge1)

#UserId = '18206299'
#merge2 = customer_recomendation(47781982)
#st.write(merge2)
#url = st.text_input('Enter CustomerID')
#st.write('The Entered Customer Id is', url)
#merge2 =merge1.loc[43173394]
#st.write(merge2)
   map_k += precisions
   mar_k += recalls
        # Get results & append algorithm name
   tmp = pd.DataFrame.from_dict(results).mean(axis=0)
   tmp = tmp.append(pd.Series(map_k / 5, index=['map_k']))
   tmp = tmp.append(pd.Series(mar_k / 5, index=['mar_k']))
   tmp = tmp.append(pd.Series([str('/Users/lalitharahul/Desktop/AutoRecommender/RecServe')], index=['data']))
   tmp = tmp.append(pd.Series([str(option)], index=['Algorithm']))

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
   #st.write(merge1.loc[merge1['UserId'] == '47781982'])
       # st.write(results_df) 
    # saving the results file to folder
#st.write('The dataset details')
#st.write(results_df))   
