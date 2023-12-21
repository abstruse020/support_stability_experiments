#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm.notebook import tqdm 
import datetime
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits, load_iris

from numpy.linalg import norm
import pickle

## Personal Imports
from my_model_e2 import FFNetwork
from db_creator import Dataset

## DETAILS
# Here we will plot our bound and the actual generalization error.
# FIX m and w, for non over parameterized setting.
# 
# SUMMARY-
# For OUR BOUND-
#   There are 3 selections- z', ith_point, random_string
#   I am doing this randomly for say t times, and then taking the mean of Lipschitz
#    constant for every epoch. And then max over all epochs to get LP and SM const.
#
# For Our bound- 
#   We need to calculate lipschitz and smoothness constant.
#     - I am using the local smoothness constant
#     - can use global i.e. need to calculate max(K_t, (f(w) - f(w'))/delta_t ) # DO IT LATER
#   We need to perform this multiple times, as we will get different ith point each time.
#    and then take the max(should be mean as expectation over r) of the lipschitz constant
#    across all experiments and (across all epochs)?, for the given K and L we will use
#    our bound to get the result. i.e we will keep changing m in our formulae and plot the
#    values we get,
#   We can do it just 50-60 times as we don't need delta_t, or loss_diff of two models.
#   IMP we shold use same order across all experiments with just ith point getting changed?
# For generalization error-
#   We calculate it using models validation loss - models train loss, we do this for every
#    step. ALso for every experiment (i.e. t times) we will do this and take the mean 
#    (or else how to plot exact as its R(S,A,r) - Re(S,A,r)) and we will get list of the 
#    generalization error as m increases.
#  
# Others-
#   Wriuting code such that we can make 2 sets


# In[92]:


use_db = 'mnist'
result_root_dir='results/t5/'
result_path='try2_diff_order'
book_keep_freq = 100
g_times = 10
g_weight = 50
g_epochs = 1
alpha_0= 0.001
exp_times_limit = 1000  ## Limit on number of random selection of z', i, and order
training_step_limit = 10000 ## this is to train for max updates per epochs
stop_hess_computation = 80 ## Stop computing hessian after calculated these many times


op_features = 5
if use_db == 'page-blocks':
    X_train, Y_train, X_val, Y_val = Dataset().get_page_blocks('datasets/page-blocks.data')
    # input_features = 10
    op_features = 5
elif use_db == 'iris':
    X_train, Y_train, X_val, Y_val = Dataset().get_iris()
    # input_features = 4
    op_features = 3
elif use_db == 'mnist':
    X_train, Y_train, X_val, Y_val = Dataset().get_mnist('datasets/mnist/')
    ## Shortening the dataset
    X_train = X_train[:20000]
    Y_train = Y_train[:20000]
    X_val   = X_val[:1000]
    Y_val   = Y_val[:1000]
    op_features = 10

def get_replaced_index_same_order(X_train):
    
    ## Forcing this function to be random ##
    # seed_num = datetime.datetime.now().microsecond
    # np.random.seed(seed_num)
    #######################################
    
    replaced_index = 0
    index = np.arange(X_train.shape[0])
    zd_index = np.random.choice(index)
    # print('index pf point to pick out:', zd_index)
    S1_ind = [i for i in index if i != zd_index]
    # print('S1_ind:',S1_ind)
    S2_ind = S1_ind
    ith_point = np.random.choice(index[:-1])
    # print('ith_point:', ith_point)
    S2_ind[ith_point] = zd_index
    # print('S2_ind:',S2_ind)
    return S1_ind, S2_ind, ith_point

def get_replaced_index(X_train):
    
    ## Forcing this function to be random ##
    seed_num = datetime.datetime.now().microsecond
    np.random.seed(seed_num)
    #######################################
    
    replaced_index = 0
    index = np.arange(X_train.shape[0])
    np.random.shuffle(index)

    S_ind = index[:-1]
    S_d_ind = np.append(index[:-2], index[-1:])
    ind_temp = np.arange(len(S_ind))
    np.random.shuffle(ind_temp)
    for i,l in enumerate(ind_temp):
        if l == len(S_ind)-1:
            replaced_index = i
            # print('replaced index :', replaced_index)
            break

    S_ind = S_ind[ind_temp]
    S_d_ind = S_d_ind[ind_temp]
    # print('S index :', S_ind)
    # print('S\' index:', S_d_ind)
    return S_ind, S_d_ind, replaced_index


# In[ ]:


def exp_once(X, Y, X_val, Y_val,weights, epochs=1, lr_rate = 1):
    
    '''
    args: X,Y,X_val, Y_val, epochs - standard inputs as the name suggest
    About: This function performs 1 experiment, It selects 1 random ith point
    creates set S1 and S2 and train models on these, we save the weights of 
    these models and calculate the norm of difference of weights (at every epoch) and append it to a file.
    
    About randomness
    Both the models use random initialization and for random_initialization = Flase, they set seed(0)
    during object creation so since both the models are called with random_initialization = False they
    both get initialized with the same weight values
    '''
    
    book_keep_grads=True
    book_keep_hessian=True
    book_keep_gen_err=True
    book_keep_weights = True
    use_model2 = False
    
    order_of_permute = []
    # To fix order of permute of both the models
    for epoch in range(epochs):
        order_of_permute.append(np.arange(X.shape[0]))
    
    S1_ind, S2_ind, ith_pos = get_replaced_index(X)
    # S1_ind, S2_ind, ith_pos = get_replaced_index_same_order(X)
    input_features = X.shape[1]
    print('ith position  :', ith_pos)
    # print('Input_features:',input_features)
    
    model1 = FFNetwork(num_hidden=weights, input_features=input_features, op_features=op_features, 
                          random_initialization=False, lr_decay=True)
    model1.fit_b(
            X=X[S1_ind], Y=Y[S1_ind], epochs=epochs, eta=lr_rate, batch_size=1,
            X_val=X_val, Y_val=Y_val, training_step_limit=training_step_limit,
            order_of_permute=order_of_permute, book_keep_freq = book_keep_freq,
            book_keep_grads=book_keep_grads, book_keep_hessian=book_keep_hessian, 
            book_keep_weights=book_keep_weights, book_keep_gen_err=book_keep_gen_err, stop_hess_computation= stop_hess_computation)
    
    if use_model2:
        model2 = FFNetwork(num_hidden=weights, input_features=input_features, op_features=op_features, 
                              random_initialization=False, lr_decay=True)
        model2.fit_b(
                X=X[S2_ind], Y=Y[S2_ind], epochs=epochs, eta=lr_rate, batch_size=1,
                X_val=X_val, Y_val=Y_val, training_step_limit=training_step_limit,
                order_of_permute=order_of_permute, book_keep_freq = book_keep_freq,
                book_keep_grads=book_keep_grads, book_keep_hessian=book_keep_hessian, 
                book_keep_weights=book_keep_weights, book_keep_gen_err=book_keep_gen_err, stop_hess_computation= stop_hess_computation)
    
    gen_err1 = model1.generalization_err
    weight1 = model1.w
    # l1 = np.array(model1.loss_list)
    # lp1 = np.array(model1.point_loss_list)
    grads1 = model1.grads_norm
    hess_norm1 = model1.hess_norms
    
    if use_model2:
        gen_err2 = model2.generalization_err
        weight2 = model2.w
        # l2 = np.array(model2.loss_list)
        # lp2 = np.array(model2.point_loss_list)
        grads2 = model2.grads_norm
        hess_norm2 = model2.hess_norms
    else:
        gen_err2 = []
        weight2 = []
        l2 = []
        lp2 = []
        grads2 = []
        hess_norm2 = []
    
    return np.array(gen_err1), np.array(gen_err2), np.array(weight1), np.array(weight2), grads1, grads2, hess_norm1, hess_norm2

def exp2_multiple_run(X, Y, X_val, Y_val, weights, times, lr_rate, root_dir='', path=None, epochs = 1, clear_files = True):
    delta_t_list = []
    gen_err_list = []
    loss_diff_list = []
    point_loss_diff_list = []
    grad_norm_list = []
    hess_norm_list = []
    gen_err_list = []
    if path is not None:
        grad_file_path = root_dir+'grad_'+path
        hess_file_path = root_dir+'hess_'+path
        gen_file_path = root_dir+'gen_'+path
    if clear_files and path is not None:
        with open(grad_file_path, 'w+') as f:
            f.write('')
        with open(hess_file_path, 'w+') as f:
            f.write('')
        with open(gen_file_path, 'w+') as f:
            f.write('')
    
    for t in range(min(times, exp_times_limit)):
        gen_err1, gen_err2, w1, w2, grads1, grads2, hess_norm1, hess_norm2 = exp_once(X,Y, X_val, Y_val,weights, epochs, lr_rate= lr_rate)
        
        grad_norm_list.append(grads1)
        hess_norm_list.append(hess_norm1)
        gen_err_list.append(gen_err1)
        
        if path is not None:
            with open(grad_file_path,'a+') as f:
                f.write(' '.join([str(grad) for grad in grads1]) + '\n')
            with open(hess_file_path,'a+') as f:
                f.write(' '.join([str(grad) for grad in hess_norm1]) + '\n')
            with open(gen_file_path,'a+') as f:
                f.write(' '.join([str(grad) for grad in gen_err1]) + '\n')
        
    return grad_norm_list, hess_norm_list, gen_err_list


# In[30]:


grad_list, hess_list, gen_err_list = exp2_multiple_run(X_train, Y_train, X_val=X_val, Y_val=Y_val, weights=g_weight, times=g_times, lr_rate=alpha_0, root_dir=result_root_dir, path=result_path, epochs=g_epochs)


# In[31]:


def get_lp_sm(grad_list, hess_list):
    mean_grads_per_update = np.mean(np.array(grad_list), axis=0)
    lp_const = np.max(mean_grads_per_update)
    
    mean_hess_per_update = np.mean(np.array(hess_list), axis=0)
    sm_const = np.max(mean_hess_per_update)
    return lp_const, sm_const

def our_bound_computatio(K_g, L_g, m, t, alpha_0, way='normal'):
    
    alpha_lg = alpha_0*L_g
    print('alpha_0 L_g (<=0.5?):',alpha_lg)
    if way == 'normal':
        F = np.power(t/m, 1/4)
        F = F*np.power(2, t/m)
        val = (1+alpha_lg)*np.power(K_g,2)/L_g
        val = val* F/(np.power(m, 1-2*alpha_lg))
    return val


# In[32]:


K_g, L_g = get_lp_sm(grad_list, hess_list)
print('Lipschitz constant:', K_g)
print('Smoothness constant:', L_g)


# In[94]:


### Saving exp details
details={}
details['Times'] = g_times
details['Dataset'] = use_db
details['book_keep_freq'] = book_keep_freq
details['Weights'] = g_weight
details['Epochs'] = g_epochs
details['alpha_0'] = alpha_0
details['Lipschitz constant'] = K_g
details['Smoothness constant'] = L_g
details['stop_hessian_computation'] = stop_hess_computation

with open(result_root_dir+'details_'+result_path+'.txt', 'w+') as f:
    for key, val in details.items():
        content = key + ' : '+str(val) + '\n'
        f.write(content)


# In[ ]:





# In[33]:


# m = np.arange(1,X_train.shape[0])
# t = m
# val = our_bound_computatio(K_g=K_g, L_g=L_g, m=m, t=t, alpha_0=alpha_0)


# In[90]:


# def plot_bound_gen_err(bound, gen_error, path=None):
#     fig, ax = plt.subplots(figsize=(8,6))
#     ax.set_yscale('log')
#     ax.set_ylim(0.001,1000)
#     ax.grid()
#     ax.plot(val, label= 'our_bound')
#     ax.plot(np.max(gen_err_list, axis=0), label='gen_err')
#     ax.set_xlabel('Number of Updates')
#     ax.set_ylabel('Generalization errors')
#     ax.legend()
#     if path is not None:
#         plt.savefig(path)


# # In[91]:


# plot_bound_gen_err(val, gen_err_list)


# In[42]:


# def plot_data(ip_list, path = None, y_label='norm'):
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ip_list = np.asarray(ip_list)
#     ip_mean = np.mean(ip_list, axis=0)
#     ip_std = np.std(ip_list, axis=0)
#     ip_min = np.min(ip_list, axis=0)
#     ip_max = np.max(ip_list, axis=0)
#     ip_90p = np.percentile(ip_list, 90, axis=0)
#     #ax.errorbar(range(len(grads_mean[:10])), grads_mean[:10],yerr= grads_std[:10], fmt='o')
#     ax.plot(ip_max, color ='black', label= 'max')
#     # ax.plot(ip_90p, 'r-', label='90th')
#     ax.set_ylabel(y_label)
#     ax.set_xlabel("No. of Update to weights")
#     ax.plot(ip_mean, 'y-', label= 'mean')
#     # ax.plot(ip_min, 'c-', label='min')
#     ax.legend()
#     if path is not None:
#         fig.savefig(path)



