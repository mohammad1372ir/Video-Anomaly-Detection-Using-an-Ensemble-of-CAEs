


import pickle as pkl

import numpy as np

import pandas as pd

import tensorflow as tf


physical_devices=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

  


with open('path_to_train_features', 'rb') as f:
    Train_Features = pkl.load(f)
    
f.close()    


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range = (0,1))

Scaled_Features = scaler.fit_transform(Train_Features)


Train_Data = np.zeros((Train_Features.shape[0]-32, 32, 768))

for i in range(32, Train_Features.shape[0]):
    
    Train_Data[i-32, :, :] = Scaled_Features[i-32:i , :]  


with open('path_to_train_features', 'rb') as f:
    Train_Features = pkl.load(f)
    
f.close()    


with open('path_to_anomaly_features', 'rb') as f:
    Anomaly_Features = pkl.load(f)
    
f.close()    
        

with open('path_to_normal_features', 'rb') as f:
    Normal_Features  = pkl.load(f)
 

f.close()



    
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range = (0,1))

Scaled_Features = scaler.fit_transform(Train_Features)

del Scaled_Features


Scaled_Normal =scaler.transform(Normal_Features)
Scaled_Anomaly =scaler.transform(Anomaly_Features)


Normal_Data = np.zeros((Normal_Features.shape[0]-32, 32, 768))

for i in range(32,Normal_Features.shape[0]):
    
    Normal_Data[i-32,:,:] = Scaled_Normal[i-32:i,:]
    

Anomaly_Data = np.zeros((Anomaly_Features.shape[0]-32, 32, 768))

for i in range(32,Anomaly_Features.shape[0]):
    
    Anomaly_Data[i-32,:,:] = Scaled_Anomaly[i-32:i,:]
    

    
    
del Scaled_Anomaly
del Scaled_Normal


num_of_experts = 10
import keras
from keras import layers

normal_bag_results = np.zeros((Normal_Data.shape[0],num_of_experts))
anomaly_bag_results = np.zeros((Anomaly_Data.shape[0],num_of_experts))
train_bag_results = np.zeros((Train_Data.shape[0] , num_of_experts))


quant = 0.75

for i in range(num_of_experts):
    

    keras.backend.clear_session()
    
    
    
    
    
    input_img = keras.Input(shape=(32, 768, 1))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = keras.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')
    

    model.load_weights('path_to_saved_caes_' +str(i) + '.hdf5')
    print(i)
    
    Normal_Recs  = model.predict(Normal_Data)
    Anomaly_Recs = model.predict(Anomaly_Data)
    
    Normal_Diffs = abs(Normal_Recs[:,:,:,0] - Normal_Data)
    Anomaly_Diffs = abs(Anomaly_Recs[:,:,:,0] - Anomaly_Data)
    
    Normal_Errs = Normal_Diffs.sum(axis = 1)
    Normal_Errs = Normal_Errs.sum(axis = 1)
    
    
    Anomaly_Errs = Anomaly_Diffs.sum(axis = 1)
    Anomaly_Errs = Anomaly_Errs.sum(axis = 1)
    
    normal_bag_results[:,i] = Normal_Errs
    anomaly_bag_results[:,i] = Anomaly_Errs
    
    
    
    Train_Recs  = model.predict(Train_Data)

    
    Train_Diffs = abs(Train_Recs[:,:,:,0] - Train_Data)

    
    Train_Errs = Train_Diffs.sum(axis = 1)
    Train_Errs = Train_Errs.sum(axis = 1)
    
    

    
    train_bag_results[:,i] = Train_Errs    
    
    

df =  pd.DataFrame({'Normal' : normal_bag_results.mean(axis = 0) , 'Anomaly' : anomaly_bag_results.mean(axis = 0) , 'Train': train_bag_results.mean(axis = 0)})


df.to_csv("test_data.csv")

    
    
    





        


    
    
    

           
        
    
    
    


 
    

