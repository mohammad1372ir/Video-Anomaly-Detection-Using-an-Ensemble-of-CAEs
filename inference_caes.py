# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:44:06 2024

@author: HP OMEN
"""



import pandas as pd
import numpy as np



amount_of_say = pd.read_csv('path_to_amount_o_say_file')


df = pd.read_csv("path_to_test_data.csv")


normal_bag_results  = df["Normal"]
anomaly_bag_results = df["Anomaly"]



err_normal = normal_bag_results
err_anomaly = anomaly_bag_results

total = np.concatenate((normal_bag_results , anomaly_bag_results), axis = 0)

MINI_ARR = total.min(axis = 0)

MAXI_ARR = total.max(axis = 0)


number_of_thresholds = 100

num_of_experts = 10

Thresholds = np.zeros((number_of_thresholds,num_of_experts))

for i in range(num_of_experts):
    Thresholds[:,i] = np.linspace(MINI_ARR[i], MAXI_ARR[i], number_of_thresholds)



Recall = np.zeros((number_of_thresholds,1))
Specificity = np.zeros((number_of_thresholds,1))

 


TP = np.zeros((number_of_thresholds,1))
TN = np.zeros((number_of_thresholds,1))
FP = np.zeros((number_of_thresholds,1))
FN = np.zeros((number_of_thresholds,1))


  

for thresh_num in range(number_of_thresholds):
    
    threshs = Thresholds[thresh_num,:]
    
    fp =0 
    tn =0
    tp =0
    fn =0
    
    for normal_idx in range(normal_bag_results.shape[0]):
    
        normal_sample = normal_bag_results[normal_idx,:]
        
        vote = 0 
        
        for expert_num in range(num_of_experts):
                        
            if normal_sample[expert_num] > threshs[expert_num] :
                
                vote = vote + amount_of_say['amount of say'].iloc[expert_num]
                
     
        if vote >= (amount_of_say['amount of say'].sum()) / 2:
            fp+=1
        else:
            tn+=1
                    

                 
                    
    
    
    for anomaly_idx in range(anomaly_bag_results.shape[0]):
    
        anomaly_sample = anomaly_bag_results[anomaly_idx,:]
        
        vote = 0 
        
        for expert_num in range(num_of_experts):
                        
            if normal_sample[expert_num] > threshs[expert_num] :
                
                vote = vote + amount_of_say['amount of say'].iloc[expert_num]
                  
        
        if vote >= (amount_of_say['amount of say'].sum()) / 2:
            tp+=1
        else:
            fn+=1
            
    xx = anomaly_bag_results.shape
    yy = normal_bag_results.shape
    
    TP[thresh_num,0] = tp/xx[0]
    TN[thresh_num,0] = tn/yy[0]
    
    
    FP[thresh_num,0] = fp/xx[0]
    FN[thresh_num,0] = fn/yy[0]
    

    Recall[thresh_num,0]      = tp/(tp+fn+.0001)
    Specificity[thresh_num,0] = tn/(tn+fp+.0001)




tpr = Recall
fpr = np.ones_like(Specificity)-Specificity
    
    
    
#%%%


from sklearn import metrics



tpr = Recall
fpr = np.ones_like(Specificity)-Specificity


AUC = metrics.auc(fpr, tpr)

print(AUC)
