import pandas as pd
import numpy as np
from scipy import sparse
import time
from concurrent.futures import ProcessPoolExecutor
import itertools

#Parses the LibFM output model, returns w0, wj, and vj,f.
#w0 is a single number, wj is an array of floats, vj,f is an array of array of floats
def parse_output_file(filename):
    w0_spot=False
    wj_spot=False
    vj_spot=False
    w0=0
    wj=[]
    vj=[]
    f = open(filename, 'r')
    for line in f:
        if 'global bias' in line:
            w0_spot=True
        elif w0_spot:
            w0=float(line)
            w0_spot=False
        elif 'unary interactions' in line:
            wj_spot=True
        elif 'pairwise interactions' in line:
            wj_spot=False
            vj_spot=True
        elif wj_spot:
            wj.append(float(line))
        elif vj_spot:
            line_split=line.split(' ')
            line_numbers = [float(i) for i in line_split]
            vj.append(line_numbers)
    return w0, np.array(wj), np.array(vj)

def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return sparse.csc_matrix((new_data, new_indices, new_ind_ptr))

def make_prediction(i, x_mat, w, V, w0):
    x = x_mat[i,:]
    x2 = (x.T).dot(x)
    x_upper = sparse.triu(x2, k=1)
    pred_inter = 0
    my_tup_upper = sparse.find(x_upper)

    rows = my_tup_upper[0]
    cols = my_tup_upper[1]
    vals = my_tup_upper[2]
    Vrows = V[rows,:]
    Vcols = V[cols,:]

    test = np.einsum('ij,ij->i', Vrows, Vcols)
    pred_inter = np.sum(np.multiply(vals, test))

    pred_linear = x.dot(w)
    prediction = (w0 + pred_linear + pred_inter)[0]
    return prediction

def make_all_predictions(v, w, w0, user, ratings, features, weight_vector):
    """Function to create predictions
    
    Input:
    v: interaction term matrix
    w: linear feature biases (array)
    w0: global bias (float)
    user: user_id we are looking at
    ratings: ratings matrix
    features: FM model features
    weight_vector: vector of weights for the genres. Should be of length 22 (one for each genre)
    
    Returns:
    predicted rating for each book, with 0 for already-rated books
    """

    # pick chosen user
    users = sparse.csc_matrix((10000, 53428), dtype=int)
    users[:,user-1] = 1

    # make matrix
    x_mat = (concatenate_csc_matrices_by_columns(users, features)).tocsr()

    # set weights
    weights = np.ones(x_mat.shape[1])
    weights[-22:] = weight_vector
    w = np.multiply(w, weights)
    V = np.multiply(v, weights.reshape(-1,1)) #transpose?

    # make predictions
    start = time.time()
    pred_mat = np.zeros(10000)
    for i in range(10000): # for each book
        pred_mat[i] = make_prediction(i, x_mat, w, V, w0)
    print(time.time() - start)

    # mask books that have already been rated
    user_rated = ratings[ratings.user_id == user].book_id.values
    mask = np.array([0 if i in user_rated else 1 for i in range(1, 10001)])

    return np.multiply(pred_mat, mask)


def get_top_k(pred_mat, books, k):
    top_ids = np.argsort(-pred_mat)[:k] + 1
    top_books = books[books.book_id.isin(top_ids)][['title', 'authors']]
    return top_books


