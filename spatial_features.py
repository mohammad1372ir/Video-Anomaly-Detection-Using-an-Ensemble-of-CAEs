

from vit_keras import vit
import tensorflow as tf
import cv2
import numpy as np

import pickle

image_size = 384

vit_model = vit.vit_b16(
        image_size = image_size,
        activation = 'sigmoid',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        )



model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),

    ],
    name = 'vision_transformer')

model.summary()

#%%%

training_adds = ["paths_to_training_frames"]

training_imgs = np.zeros((len(training_adds),384,384,3))

cnt = 0
for add in training_adds:
        
    img = cv2.imread(add)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(384,384))
    training_imgs[cnt,:,:,:] = img
    cnt+=1
    
train_data = model.predict(training_imgs)


with open('path_to_training_spatial_features.pkl','wb') as f:
    pickle.dump(train_data, f)
    
#%%%


normal_adds = ["paths_to_normal_frames"]

normal_imgs = np.zeros((len(normal_adds),384,384,3))

cnt = 0
for add in normal_adds:
        
    img = cv2.imread(add)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(384,384))
    normal_imgs[cnt,:,:,:] = img
    cnt+=1
    
normal_data = model.predict(normal_imgs)

with open('path_to_normal_spatial_features.pkl','wb') as f:
    pickle.dump(normal_data, f)    

#%%%
    
anomaly_adds = ["paths_to_anomaly_frames"]

anomaly_imgs = np.zeros((len(anomaly_adds),384,384,3))

cnt = 0
for add in anomaly_adds:
        
    img = cv2.imread(add)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(384,384))
    anomaly_imgs[cnt,:,:,:] = img
    cnt+=1
    
anomaly_data = model.predict(anomaly_imgs)


with open('path_to_anomaly_spatial_features.pkl','wb') as f:
    pickle.dump(anomaly_data, f)           
