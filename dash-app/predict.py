import pandas as pd
import numpy as np
from scipy import sparse
import time

# books = pd.read_csv('../part2/books.csv')

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

# w0, wj, vj = parse_output_file('model1.libfm')

# num_col=len(wj)

# wj = np.array(wj)
# vj = np.array(vj)

# books_genres = sparse.load_npz('books_genres.npz')

def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return sparse.csc_matrix((new_data, new_indices, new_ind_ptr))

def output_top_k(v, w, w0, k, user, book_genres, weight_vector):
    """Function to create predictions
    Input:
    V: interaction term matrix
    w: linear feature biases (array)
    w0: global bias (float)
    k: number of books we want to recommend
    user: user_id we are looking at
    book_genres: matrix of books one hot encoded and genres in csc sparse format
    weight_vector: vector of weights for the genres. Should be of length 22 (one for each genre)
    scaler: by what factor we want to scale the genre weights
    Returns:
    top k list"""

    # start_time = time.time()

    # pick chosen user
    users = sparse.csc_matrix((10000, 53425), dtype=int)
    users[:,user] = 1

    # make matrix
    x_mat = (concatenate_csc_matrices_by_columns(users, book_genres)).tocsr()

    # set weights
    weights = np.ones(x_mat.shape[1])
    w = np.multiply(w, weights)
    V = np.multiply(v, weights.reshape(-1,1)) #transpose?

    # make predictions
    pred_mat = np.zeros(10000)
    for i in range(10000): # for each book
        x = x_mat[i,:]
        x2 = (x.T).dot(x)
        x_upper = sparse.triu(x2, k=1)
        pred_inter = 0
        my_tup_upper = sparse.find(x_upper)
        for entry in range(my_tup_upper[0].shape[0]):
            row = my_tup_upper[0][entry]
            col = my_tup_upper[1][entry]
            value = my_tup_upper[2][entry]
            pred_inter += value*((V[row,:]).dot(V[col,:]))
        pred_linear = x.dot(w)
        pred_mat[i] = w0 + pred_linear + pred_inter
    
    # return top k
    top_ids = np.argsort(-pred_mat)[:k]
    return top_ids

def get_book_info(top_ids, books):
    # top_books = []
    # for i in top_ids:
        # top_books.append(books.loc())
    top_books = books[books.book_id.isin(top_ids)][['title', 'authors']]
    # print(time.time() - start_time)
    return top_books
