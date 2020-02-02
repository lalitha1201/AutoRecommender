import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# data = pd.read_csv('data_subset.csv')
#
# # data formatted to be able to use surprise package
# data_surprise = data[['customer_id', 'product_id','star_rating']].\
#     rename(columns={'customer_id': 'userID', 'product_id': 'itemID', 'star_rating': 'rating'})
#
# data_surprise.to_csv('data_surprise.csv', index=False)

def get_data(_file_path):
    """
    Method to generate a clean pandas dataframe given file path
    :param _file_path: input raw data file path
    :param _save_path: output data_subset file path
    :return: data_subset dataframe
    """
    # reading the data from tsv
    cols = pd.read_csv(_file_path, sep='\t', nrows=1).columns
    df = pd.read_csv(_file_path, sep='\t', usecols=cols)

    # selecting a subset of the data with customers who purchased min 10 items and items which have a min of 10 ratings
    # df_dense = df.groupby('product_id').filter(lambda x: x['customer_id'].nunique() >= 10).reset_index()
    # df_denser = df_dense.groupby('customer_id').filter(lambda x: x['product_id'].nunique() >= 10).reset_index()

    data_subset = df.sample(frac=0.1)[['customer_id', 'product_id', 'product_parent',
                                       'product_title', 'star_rating', 'review_date']]

    # changing customer_id, product_parent to object
    # star rating to int, review_date to date
    data_subset['customer_id'] = data_subset['customer_id'].astype(str)
    data_subset['product_parent'] = data_subset['product_parent'].astype(str)
    data_subset = data_subset[~data_subset['star_rating'].astype(str).str.startswith('20')]
    data_subset['star_rating'] = data_subset['star_rating'].astype(float, errors='ignore')
    data_subset['review_date'] = data_subset['review_date'].apply(lambda x: pd.to_datetime(x, errors='coerce',
                                                                                           format='%Y-%m-%d'))
    # saving the cleaned file to folder
#    if _save_path:
    data_subset.to_csv('amazon_prep.csv', index=False)

    return data_subset
