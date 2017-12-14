
# coding: utf-8

# In[ ]:

#53424 users 53425 book is 2757 or 2756
#22 final genres


# In[1]:

import pandas as pd
import numpy as np
from scipy import sparse
import time


# In[63]:

books = pd.read_csv('books.csv')


# In[2]:

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
    return w0,wj,vj


# In[3]:

w0, wj, vj = parse_output_file('../libfm_writes/model1.libfm')


# In[4]:

num_col=len(wj)


# In[39]:

#input_vector = np.array(input_vector)
wj = np.array(wj)
vj = np.array(vj)


# ### Testing with 10k books and given user

# In[5]:

books_genres = sparse.load_npz('books_genres.npz')


# In[23]:

def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return sparse.csc_matrix((new_data, new_indices, new_ind_ptr))


# ### Own Sparse Creation

# In[187]:

def output_top_k(V, w, w0, k, user, book_genres, weight_vector, scaler):
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
    start_time = time.time()
    users = sparse.csc_matrix((10000, 53425), dtype=int)
    users[:,user] = 1
    x_mat = (concatenate_csc_matrices_by_columns(users, book_genres)).tocsr()
    # weights for entire matrix
    weights = np.ones(x_mat.shape[1])
    weights[-40:] = scaler*weight_vector
    w = np.multiply(w, weights)
    V = np.multiply(V.T, weights)
    pred_mat = np.zeros(10000)
    for i in range(10000):
        x = x_mat[i,:]
        x2 = (x.T).dot(x)
        x_upper = sparse.triu(x2, k=1)
        pred_inter = 0
        my_tup_upper = sparse.find(x_upper)
        for entry in range(my_tup_upper[0].shape[0]):
            row = my_tup_upper[0][entry]
            col = my_tup_upper[1][entry]
            value = my_tup_upper[2][entry]
            pred_inter += value*((vj[row,:]).dot(vj[col,:]))
        pred_linear = x.dot(w)
        pred_mat[i] = w0 + pred_linear + pred_inter
    
    # returning top k
    top_ids = np.argsort(-pred_mat)[:k]
    top_books = []
    for i in top_ids:
        top_books.append(books.loc())
    top_books = books[books.book_id.isin(top_ids)]['title']
    print(time.time() - start_time)
    return top_books


# In[188]:

weight_vector = np.ones(40)
weight_vector[33] = 50.0
scaler=1
my_preds = output_top_k(vj,wj,w0, 5, 42652, books_genres.tocsc(), weight_vector, scaler)


# In[189]:

print(my_preds)


# ### Testing shit

# In[37]:

#Given a string (line from the libfm testing set), outputs an input vector
def create_input_vector(sample, num_col): 
    input_vector=[0]*num_col
    sample_split = sample.split(' ')
    sample_split = sample_split[1:]
    for s in sample_split:
        col_val = int(s[:s.index(':')])
        item_val = float(s[s.index(':')+1:])
        input_vector[col_val]=item_val
    return input_vector


# In[38]:

sample = '3 42562:1 56182:1 63432:0.01118210862619808 63434:0.02396166134185303 63439:0.2220447284345048 63442:0.7092651757188498 63443:0.007987220447284345 63456:0.0255591054313099'
input_vector = create_input_vector(sample, num_col)
#4.61783


# In[35]:

def make_prediction_2(x, V, w, w0):
    """Function to create predictions
    Input:
    x: feature vector
    V: interaction term matrix
    w: linear feature biases (array)
    w0: global bias (float)
    Returns:
    predicted rating (float)"""
    start_time = time.time()
    for i in range(10000):
#        print(i)
        x = sparse.csr_matrix(x)
        x2 = (x.T).dot(x)
        x_upper = sparse.triu(x2, k=1)
        pred_inter = 0
        my_tup_upper = sparse.find(x_upper)
        for entry in range(my_tup_upper[0].shape[0]):
        #    print(entry)
            i = my_tup_upper[0][entry]
            j = my_tup_upper[1][entry]
            value = my_tup_upper[2][entry]
            pred_inter += value*((vj[i,:]).dot(vj[j,:]))
        pred_linear = x.dot(w)
    print(time.time() - start_time)
    return w0 + pred_linear + pred_inter


# #### Weight matrix

# In[144]:

np.ones(40).shape


# In[190]:

w_test = [1, 1, 1, 1, 1, 1, 1]
weight_test = np.ones(7)
weight_test[-3:] = [1,2,4]
a = [1,2,3,4,5]
V = np.array((a, a, a, a, a, a, a))


# In[191]:

np.multiply(V.T, weight_test)


# In[137]:

print(2*weight_test)


# In[132]:

print(V)


# In[37]:

vj[2].dot(vj[4])


# In[38]:

wj.dot(input_vector)


# In[39]:

input_vector.shape


# In[41]:

x2 = (x.T).dot(x)


# In[42]:

x2.shape


# In[43]:

my_tup = sparse.find(x2)


# In[44]:

my_tup


# In[45]:

x_upper = sparse.triu(x2, k=1)


# In[46]:

my_tup_upper = sparse.find(x_upper)


# In[47]:

my_tup_upper[0].shape[0]


# In[48]:

print(my_tup_upper[0][3])
print(my_tup_upper[1][3])
print(my_tup_upper[2][3])


# In[49]:

vj[10714].dot(vj[63437])*0.01399


# In[50]:

my_tup[2].shape[0]


# In[51]:

my_tup[0][0]


# In[52]:

vj[i].shape


# ### Attempting with trick from paper

# In[75]:

vx = input_vector.dot(vj[:,0])


# In[77]:

vx_sq = np.square(input_vector).dot(np.square(vj[:,0]))


# In[82]:

vx_sq**2


# In[80]:

k = vj.shape[1]


# In[13]:

def predict_fm(x, w0, w, V):
    start_time = time.time()
    for i in range(10000):
        my_sum = 0
        for f in range(vj.shape[1]):
            vx = (x.dot(vj[:,f]))**2
            vx_sq = np.square(x).dot(np.square(vj[:,f]))
            my_sum += (vx - vx_sq)
        interaction_term = my_sum/2.0
        linear_term = w.dot(x)
    print(time.time() - start_time)
    return w0 + linear_term + interaction_term


# In[33]:

def predict_fm_2(x, w0, w, V):
    start_time = time.time()
    for i in range(10000):
        print(i)
        vx = (vj.T).dot(x)
        vx_sq = np.square(vj.T).dot(x.power(2))
        interaction_term = (vx - vx_sq).sum()/2
        linear_term = w.dot(x)
    print(time.time() - start_time)
    return w0 + linear_term + interaction_term


# In[34]:

predict_fm_2(sparse.csr_matrix(input_vector), w0, wj, vj)


# In[30]:

sparse_x = sparse.csr_matrix(input_vector)
sparse_x.shape


# In[15]:

vj.shape


# In[29]:

vj.shape[1]


# In[88]:

linear_term = wj[:,0].dot(input_vector)


# In[89]:

linear_term


# In[90]:

w0 + linear_term + interaction_term


# In[91]:

input_vector


# In[ ]:



