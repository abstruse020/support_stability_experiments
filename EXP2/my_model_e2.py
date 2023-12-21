import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
#import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm.notebook import tqdm 
#import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler, MinMaxScaler
#from sklearn.datasets import load_iris

from numpy.linalg import norm

class FFNetwork:
  
    def __init__(self, num_hidden, input_features, op_features, init_method = 'xavier', activation_function = 'relu', leaky_slope = 0.1, random_initialization = False, lr_decay = True):
        
        self.ran_fit = False
        self.params={}
        self.num_layers=2
        self.layer_sizes = [input_features, num_hidden, op_features]
        self.classes = range(op_features)
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope
        self.lr_decay = lr_decay
        if not random_initialization:
            # np.random.seed(0)
            self.seed_values = np.random.RandomState(seed=10).randint(100000, size = (self.num_layers,2))
        else:
            self.seed_values = np.random.randint(100000, size = (self.num_layers,2))
        # print('seed values size:',self.seed_values.shape)
        if init_method == "random":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)] = np.random.RandomState(seed=self.seed_values[i-1,0]).randn(self.layer_sizes[i-1],self.layer_sizes[i])
                self.params["B"+str(i)] = np.random.RandomState(seed=self.seed_values[i-1,1]).randn(1,self.layer_sizes[i])

        elif init_method == "he":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)] = np.random.RandomState(seed=self.seed_values[i-1,0]).randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(2/self.layer_sizes[i-1])
                self.params["B"+str(i)] = np.random.RandomState(seed=self.seed_values[i-1,1]).randn(1,self.layer_sizes[i])

        elif init_method == "xavier":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)]=np.random.RandomState(seed=self.seed_values[i-1,0]).randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(1/self.layer_sizes[i-1])
                self.params["B"+str(i)]=np.random.RandomState(seed=self.seed_values[i-1,1]).randn(1,self.layer_sizes[i])

        self.gradients={}
        self.update_params={}
        self.prev_update_params={}
        self.enc = None
        for i in range(1,self.num_layers+1):
            self.update_params["v_w"+str(i)]=0
            self.update_params["v_b"+str(i)]=0
            self.update_params["m_b"+str(i)]=0
            self.update_params["m_w"+str(i)]=0
            self.prev_update_params["v_w"+str(i)]=0
            self.prev_update_params["v_b"+str(i)]=0
  
    def forward_activation(self, X): 
        if self.activation_function == "sigmoid":
            return 1.0/(1.0 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        elif self.activation_function == "relu":
            return np.maximum(0,X)
        elif self.activation_function == "leaky_relu":
            return np.maximum(self.leaky_slope*X,X)

    def grad_activation(self, X):
        if self.activation_function == "sigmoid":
            return X*(1-X) 
        elif self.activation_function == "tanh":
            return (1-np.square(X))
        elif self.activation_function == "relu":
            return 1.0*(X>0)
        elif self.activation_function == "leaky_relu":
            d=np.zeros_like(X)
            d[X<=0]=self.leaky_slope
            d[X>0]=1
            return d

    def get_accuracy(self, X, Y):    
        Y_pred_train = self.predict(X)
        Y_pred_train = np.argmax(Y_pred_train,1)
        accuracy_train = accuracy_score(Y_pred_train, Y)
        return accuracy_train
    
    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1,1)
  
    def forward_pass(self, X, params = None):
        if params is None:
            params = self.params
        self.A1 = np.matmul(X, params["W1"]) + params["B1"] # (N, 2) * (2, 2) -> (N, 2)
        self.H1 = self.forward_activation(self.A1) # (N, 2)
        self.A2 = np.matmul(self.H1, params["W2"]) + params["B2"] # (N, 2) * (2, 2) -> (N, 2)
        self.H2 = self.softmax(self.A2) # (N, 2)
        return self.H2

  
    def grad(self, X, Y, params = None):
        if params is None:
            params = self.params 

        self.forward_pass(X, params)
        m = X.shape[0]
        self.gradients["dA2"] = self.H2 - Y # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"]) # (2, N) * (N, 4) -> (2, 4)
        self.gradients["dB2"] = np.sum(self.gradients["dA2"], axis=0).reshape(1, -1) # (N, 4) -> (1, 4)
        self.gradients["dH1"] = np.matmul(self.gradients["dA2"], params["W2"].T) # (N, 4) * (4, 2) -> (N, 2)
        self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(self.H1)) # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dW1"] = np.matmul(X.T, self.gradients["dA1"]) # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dB1"] = np.sum(self.gradients["dA1"], axis=0).reshape(1, -1) # (N, 2) -> (1, 2)
    
    def book_keeping_grads(self, gradw, gradb, batch=None, epoch=None):
        gw1d = []
        gb1d = []
        for w in gradw:
            gw1d.extend(w.reshape(-1))
        for b in gradb:
            gb1d.extend(b.reshape(-1))
        #print('W',len(gw1d))
        #print('b',len(gb1d))
        norm_w = np.linalg.norm(gw1d)
        norm_b = np.linalg.norm(gb1d)
        gw1d.extend(gb1d)
        norm_wb = np.linalg.norm(gw1d)
        self.grads_norm.append(norm_wb)
    
    def book_keeping_weights(self, weights, biases, batch=None, epoch=None):
        w=[]
        b=[]
        for w_i in weights:
            w.extend(w_i.reshape(-1))
        for b_i in biases:
            b.extend(b_i.reshape(-1))
        w.extend(b)
        self.w.append(w)
        
    def book_keep_generalization_error(self, loss_train, loss_val):
        self.generalization_err.append(loss_train - loss_val)
        
    def model_gradients(self, X, Y, params, display_logs=False, eps_0 = None):
        X = X.astype(float)
        N = X.shape[0]
        gradient = []
        op_0     = self.get_loss(X, Y, params)
        times    = 0
        zero_params_count = 0
        zero_eps_count    = 0
        
        # To iterate through all params
        for key, W in self.params.items():
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    times+=1
                    if W[i][j] == 0:
                        zero_params_count +=1
                    if eps_0 is None:
                        eps_0 = abs(W[i][j])
                    eps = eps_0 * np.finfo(np.float32).eps
                    if eps == 0:
                        zero_eps_count +=1
                    W_ij = 1. * W[i][j]
                    W[i][j] = W[i][j] + eps
                    op_1 = self.get_loss(X, Y, params)
                    gradient.append((op_1 - op_0)/eps)
                    W[i][j] = W_ij
        if display_logs:
            print('zero params count', zero_params_count)
            print('zero eps count', zero_eps_count)
            print('times', times)
        return np.array(gradient)
    
    def get_gradients_vector(self, batch_size = None, l2_norm=False, lambda_val=0.8):
        gradw = []
        gradb = []
        gw1d  = []
        gb1d  = []
        if batch_size is None:
            batch_size = self.batch_size
        
        for i in range(1,self.num_layers+1):
            if l2_norm:
                gradw.append(lambda_val * self.params["W"+str(i)]/batch_size + self.gradients["dW" + str(i)]/batch_size)
            else:
                gradw.append(self.gradients["dW" + str(i)]/batch_size)
            gradb.append(self.gradients["dB" + str(i)]/batch_size)
        
        for w in gradw:
            gw1d.extend(w.reshape(-1))
        for b in gradb:
            gb1d.extend(b.reshape(-1))
        gw1d.extend(gb1d)
        return np.asarray(gw1d)
    
    def fast_model_grad(self, X, y_OH, params):
        self.grad(X, y_OH, params)
        gradients = self.get_gradients_vector()
        return gradients
    
    def fast_hessian(self, X, y_OH, params):
        gd_0 = self.fast_model_grad(X, y_OH, params)
        N = len(gd_0)
        hessian = np.zeros((N,N))
        eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps 
        # print('gd_0', gd_0)
        # print('eps', eps)
        if eps == 0:
            esp = np.finfo(np.float32).eps
            print("################## EPS zero in hessian computation ################")

        zero_params_count = 0
        times = 0
        # To iterate through all params
        for key, W in self.params.items():
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    times+=1
                    if W[i][j] == 0:
                        zero_params_count +=1
                    
                    W_ij = 1. * W[i][j]
                    W[i][j] = W[i][j] + eps
                    gd_1 = self.fast_model_grad(X, y_OH, params)
                    hessian[:,i] = ((gd_1 - gd_0)/eps).reshape(N)
                    W[i][j] = W_ij
        return hessian
    
    def book_keeping_hessian(self, X, y_OH, params, hessian_ord=2):
        hessian = self.fast_hessian(X, y_OH, params)
        hess_norm = np.linalg.norm(hessian, ord=hessian_ord)
        # self.hessian_list.append(hessian)
        self.hess_norms.append(hess_norm)

        
    def fit_b(self, X, Y, epochs=1, algo= "GD",l2_norm=False, lambda_val=0.8, display_logs=False, eta=1, batch_size=1, order_of_permute = None,X_val=None, Y_val = None, book_keep_grads=False, book_keep_hessian=False, book_keep_weights=False, book_keep_gen_err=False, training_step_limit = 100000, book_keep_freq = 10, stop_hess_computation = 50):
        
        ########## Initialization ##########
        self.ran_fit = True
        self.train_accuracies={}
        self.val_accuracies  ={}
        
        self.enc = OneHotEncoder()
        y_OH_train = self.enc.fit_transform(np.expand_dims(Y,1)).toarray()
        if Y_val is not None:
            y_OH_val = self.enc.transform(np.expand_dims(Y_val,1)).toarray()
        
        m = X.shape[0]
        batches = int(m/batch_size)
        self.batch_size = batch_size
        
        self.layer_sizes[0]  = X.shape[1]
        self.layer_sizes[-1] = y_OH_train.shape[1]
        # print('layer sizes', self.layer_sizes)
        self.classes = range(y_OH_train.shape[1])
        
        
        self.grads_norm      = []
        self.w               = []
        self.hess_norms      = []
        self.hessian_list    = []
        self.generalization_err = []
        self.loss_list       = []
        self.point_loss_list = []
        hessian_computation_count = 0
        # self.grads_norm_appx = []
        
        ########## Checks ###########
        if order_of_permute is not None:
            assert len(order_of_permute) >= epochs, "More #epochs than permutation order for every epochs"
        
        ########## Training starts ##########
        lr_decay_denominator = 0
        terminate_learning = False
        for num_epoch in tqdm(range(epochs), total=epochs, unit="epoch", disable=True):
            if terminate_learning:
                break
            
            if order_of_permute is None:
                ind     = np.arange(m)
                np.random.shuffle(ind)
            else:
                ind = np.asarray(order_of_permute[num_epoch])
            #for batch_no in range(batches):
            for batch_no in tqdm(range(batches), total=batches, unit="update"):
                
                if batch_no > training_step_limit:
                    terminate_learning = True
                    break
                
                if self.lr_decay:
                    lr_decay_denominator+=1
                    lr_local = eta/lr_decay_denominator
                else:
                    lr_local = eta
                    
                pick_batch = ind[batch_no*batch_size : (batch_no +1)*batch_size]
                # print('X batch size',X[pick_batch].shape)
                self.grad(X[pick_batch], y_OH_train[pick_batch])
                
                ########## For Book Keeping ##########
                gradw_per_update = []
                gradb_per_update = []
                w_per_update = []
                b_per_update = []
                ########## END ##########
                
                for i in range(1,self.num_layers+1):
                    if batch_no%book_keep_freq == 0:
                        w_per_update.append(self.params["W"+str(i)])
                        b_per_update.append(self.params["B"+str(i)])
                    if l2_norm:
                        self.params["W"+str(i)] -= (lr_local * lambda_val)/ batch_size * self.params["W"+str(i)] + lr_local * (self.gradients["dW"+str(i)]/batch_size)
                        if batch_no % book_keep_freq == 0:
                            gradw_per_update.append(lambda_val*self.params["W"+str(i)]/batch_size + self.gradients["dW" + str(i)]/batch_size)
                    else:
                        self.params["W"+str(i)] -= lr_local * (self.gradients["dW"+str(i)] /batch_size)
                        if batch_no % book_keep_freq == 0:
                            gradw_per_update.append(self.gradients["dW" + str(i)]/batch_size)
                    self.params["B"+str(i)] -= lr_local * (self.gradients["dB"+str(i)]/batch_size)
                    
                    if batch_no % book_keep_freq == 0:
                        gradb_per_update.append(self.gradients["dB" + str(i)]/batch_size)
                        
                if book_keep_grads and (batch_no%book_keep_freq == 0):
                    self.book_keeping_grads(gradw_per_update, gradb_per_update)
                if book_keep_hessian and (batch_no%book_keep_freq == 0):
                    if hessian_computation_count < stop_hess_computation:
                        hessian_computation_count +=1
                        self.book_keeping_hessian(X[pick_batch], y_OH_train[pick_batch], self.params)
                if book_keep_weights and (batch_no%book_keep_freq == 0):
                    self.book_keeping_weights(w_per_update, b_per_update)
                if book_keep_gen_err and (batch_no%book_keep_freq == 0):
                    if Y_val is None:
                        raise Exception("Provide X_val and Y_val to calculate generalization error")
                    loss_train = self.get_loss(X, Y)
                    loss_val   = self.get_loss(X_val, Y_val)
                    point_loss = self.get_loss(X[pick_batch], Y[pick_batch])
                    self.loss_list.append(loss_val)
                    self.point_loss_list.append(point_loss)
                    self.book_keep_generalization_error(loss_train, loss_val)
            
            if display_logs:
                train_accuracy=self.get_accuracy(X,Y)
                self.train_accuracies[num_epoch]=train_accuracy
                if Y_val is not None:
                    val_accuracy = self.get_accuracy(X_val, Y_val)
                    self.val_accuracies[num_epoch]=val_accuracy
        
        ########## Training Ends ##########
        if display_logs:
            print('Train accuracies', self.train_accuracies.values())
        
    
    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()
    
    def predict_with_params(self, X, params= None):
        Y_pred = self.forward_pass(X, params)
        Y_pred = np.array(Y_pred).squeeze()
        # print('y shape', Y_pred.shape)
        return np.argmax(Y_pred, 1)
    
    def get_loss(self, X, Y, params=None):
        Y_pred = self.forward_pass(X, params)
        Y_pred = np.array(Y_pred)
        if self.classes == []:
            op_labels = range(Y_pred.shape[1])
        else:
            op_labels = self.classes
        loss = log_loss(Y, Y_pred, labels=op_labels)
        return loss