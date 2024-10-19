


import numpy as np
import keras
from keras import layers


import tensorflow as tf


physical_devices=tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)




import pickle as pkl

with open('path_to_spatial_train_features', 'rb') as f:
    Features = pkl.load(f)    
    
    
    
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range = (0,1))

Scaled_Features = scaler.fit_transform(Features)


Train_Data = np.zeros((Features.shape[0]-32, 32, 768))

for i in range(32,Features.shape[0]):
    
    Train_Data[i-32,:,:] = Scaled_Features[i-32:i,:]





#%%%

#weights init and cdf

Weighted_Data = Train_Data.copy()


    
    
    
    
#%%%
 

opt = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07)
   
    
quant = 0.75

num_of_experts = 10
epochs = 50
batch_size = 40

Amount_of_Says = []

Data = dict()

Data['error_threshs'] = []
Data['err_indexes']    = []
Data['cor_indexes']    = []
Data['samples']       = []
Data['error_thresh_alters'] = []
Data['errs'] = []

for i in range(num_of_experts):
    
    
    
    
    Weights_Arr = np.ones((Weighted_Data.shape[0],1))/Weighted_Data.shape[0]

    Weights_Arr_CDF = np.zeros_like(Weights_Arr)
    
    
    
    
    for ii in range(Weights_Arr.shape[0]):
        
        Weights_Arr_CDF[ii] = Weights_Arr[:ii].sum()
    

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
    model.compile(optimizer=opt, loss='mse')
    
    model.fit(Weighted_Data , Weighted_Data , epochs = epochs , batch_size= batch_size , validation_split=0.1)
    
    
    
    model.save_weights('path_to_save_caes_' +str(i) + '.hdf5')
    
    
    Train_Recs = model.predict(Train_Data)



    Train_Diffs = abs(Train_Recs[:,:,:,0] - Train_Data)
    
    
    Train_Errs = Train_Diffs.sum(axis = 1)
    Train_Errs = Train_Errs.sum(axis = 1)


    
    error_thresh = np.quantile(Train_Errs, quant)   
    error_thresh_alter = 0.6 * Train_Errs.max()  
    
    err_indexs = np.where(Train_Errs>=error_thresh)
    cor_indexs = np.where(Train_Errs<=error_thresh)
    
    
    
    err = Weights_Arr[err_indexs].sum()
    
    amount_of_say = 0.5 * np.log((1-err)/err)
    
    Amount_of_Says.append(amount_of_say)
    
    
    Weights_Arr[err_indexs] = Weights_Arr[err_indexs] * np.e**(amount_of_say)
    Weights_Arr[cor_indexs] = Weights_Arr[cor_indexs] * np.e**((-1*amount_of_say))
    
    Weights_Arr = Weights_Arr/Weights_Arr.sum()
    
    for ii in range(Weights_Arr.shape[0]):
    
        Weights_Arr_CDF[ii] = Weights_Arr[:ii].sum()
        
    
    samples = np.random.random_sample(Train_Data.shape[0])
    
    
    Data['error_threshs'].append(error_thresh)
    Data['err_indexes'].append(err_indexs)
    Data['cor_indexes'].append(cor_indexs)
    Data['errs'].append(err)
    Data['samples'].append(samples)
    Data['error_thresh_alters'].append(error_thresh_alter)
    
    
    Temp_Data = np.zeros_like(Train_Data)
    
    for ii in range(len(samples)):
        
        Temp_Data[ii,:,:] = Train_Data[np.where(Weights_Arr_CDF <=samples[ii])[0][-1],:,:].copy()   
        
    
    Weighted_Data = Temp_Data.copy()
    
    
    
    
    print(i)
    
    
    

    
import pandas as pd


DF = pd.DataFrame({"amount of say":Amount_of_Says})

DF.to_csv('amount_of_say.csv')
