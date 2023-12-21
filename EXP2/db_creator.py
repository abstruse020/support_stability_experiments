import pandas as pd
import numpy as np
import struct as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits, load_iris


class Dataset():
    
    def __init__(self):
        
        return None
    
    def scale_dataset(self, X_train, X_val):
        scalar = MinMaxScaler()
        X_train = scalar.fit_transform(X_train)
        X_val = scalar.transform(X_val)
        return X_train, X_val
    
    def get_iris(self, scale_data=True):
        iris = load_iris()
        data, target = iris.data, iris.target
        X_train, X_val, Y_train, Y_val = train_test_split(data, target, test_size=0.33, random_state=1)
        print('X,Y train', X_train.shape, Y_train.shape)
        print('X,Y val', X_val.shape, Y_val.shape)
        if scale_data:
            X_train, X_val = self.scale_dataset(X_train, X_val)
        return X_train, Y_train, X_val, Y_val
    
    def get_page_blocks(self, path, scale_data = True):
        df = pd.read_csv(path ,delim_whitespace=True, header=None)
        print(df.shape)
        data = np.array(df.iloc[:,:10])
        target = np.array(df[10])
        print('input shape',data.shape)
        print('target shape',target.shape)
        print('uniq target', df[10].unique())
        X_train, X_val, Y_train, Y_val = train_test_split(data, target, test_size=0.33, random_state=1)
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        X_val = np.asarray(X_val)
        Y_val = np.asarray(Y_val)
        print('X,Y train', X_train.shape, Y_train.shape)
        print('X,Y val', X_val.shape, Y_val.shape)
        if scale_data:
            X_train, X_val = self.scale_dataset(X_train, X_val)
        return X_train, Y_train, X_val, Y_val
    
    def get_mnist(self, path, scale_data = True):
        train_filename = {'images' : path+'train-images-idx3-ubyte' ,
                          'labels' : path+'train-labels-idx1-ubyte'}
        with open(train_filename['images'],'rb') as train_imagesfile:
            train_imagesfile.seek(0)
            magic = st.unpack('>4B',train_imagesfile.read(4))
            nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
            nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
            nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column
            images_array = np.zeros((nImg,nR,nC))
            nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
            images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR*nC))

        with open(train_filename['labels'], 'rb') as train_labelfile:
            train_labelfile.seek(0)
            magic = st.unpack('>4B', train_labelfile.read(4))
            nItm  = st.unpack('>I', train_labelfile.read(4))[0]
            labels_array = np.zeros(nItm)
            # print('nItm:',nItm)
            labels_array = np.asarray(st.unpack('>'+'B'*nItm, train_labelfile.read(nItm)))
            
        X_train, X_val, Y_train, Y_val = train_test_split(images_array, labels_array, test_size=0.33, random_state=1)
        if scale_data:
            X_train, X_val = self.scale_dataset(X_train, X_val)
            
        print('X,Y train', X_train.shape, Y_train.shape)
        print('X,Y val', X_val.shape, Y_val.shape)
        return X_train, Y_train, X_val, Y_val
    
