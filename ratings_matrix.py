import pandas as pd

data = pd.read_csv('data_subset.csv')


def matrix_sparsity(data):
    # ratings matrix
    ratings_matrix = data.pivot_table(index=['product_id'],
                                      columns=['customer_id'],
                                      values='star_rating').fillna(0)

    # calcuate total number of entries in the item-user matrix
    num_entries = ratings_matrix.shape[0] * ratings_matrix.shape[1]

    # calculate total number of entries with zero values
    num_zeros = (ratings_matrix == 0).sum(axis=1).sum()

    # calculate ratio of number of zeros to number of entries
    sparsity = (1 - num_zeros / num_entries) * 100
    print('Sparsity of the data is {:.2%}'.format(sparsity))
    return sparsity
