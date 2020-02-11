import pandas as pd


def curated_data(_file_path, _save_path):
    # low customer ct dataset
    data = pd.read_csv(_file_path)
    df_dense = data.groupby('product_id').filter(lambda x: x['customer_id'].nunique() >= 2).reset_index()
    df_denser = df_dense.groupby('customer_id').filter(lambda x: x['product_id'].nunique() >= 5).reset_index()
    df_denser.to_csv(_save_path + '_low_users.csv', index=False)

    # High density dataset
    df_2 = data.groupby('product_id').filter(lambda x: x['customer_id'].nunique() >= 6).reset_index()
    df_1 = df_2.groupby('customer_id').filter(lambda x: x['product_id'].nunique() >= 3).reset_index()
    df_1.to_csv(_save_path + '_denser.csv', index=False)

    # Low std dataset
    df_3 = data[data.star_rating > 2]
    df_3 = df_3[data.star_rating < 5]
    df_3.to_csv(_save_path + '_lowstd.csv', index=False)

    # High std dataset
    df_4 = data[data.star_rating == 1]
    df_5 = data[data.star_rating == 5].sample(frac=0.1)
    df_5 = df_5.append(df_4)
    df_5.to_csv(_save_path + '_highstd.csv', index=False)

    # low product ct, high density dataset
    df_dense = data.groupby('product_id').filter(lambda x: x['customer_id'].nunique() >= 15).reset_index()
    df_dense.to_csv(_save_path + '_lowitem_dense.csv', index=False)

    # low customer ct, high density dataset
    df_dense1 = data.groupby('customer_id').filter(lambda x: x['product_id'].nunique() >= 10).reset_index()
    df_dense1.to_csv(_save_path + '_lowuser_dense.csv', index=False)

    # ow product ct, high density+ dataset
    df_denser = df_dense.groupby('customer_id').filter(lambda x: x['product_id'].nunique() >= 3).reset_index()
    df_denser.to_csv(_save_path + '_lowitem_denser.csv', index=False)

    # low customer ct, high density dataset
    df_denser1 = df_dense1.groupby('product_id').filter(lambda x: x['customer_id'].nunique() >= 3).reset_index()
    df_denser1.to_csv(_save_path + '_lowuser_denser.csv', index=False)

    return
