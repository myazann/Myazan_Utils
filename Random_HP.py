#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import xgboost
import lightgbm
from sklearn.tree import *
from sklearn.metrics import *
from sklearn.ensemble import *


# ## Decision Tree

# In[ ]:


class_weight = [{0:1,1:4},{0:1,1:5},{0:1,1:6},{0:1,1:7},{0:1,1:8},{0:1,1:9},{0:1,1:10}]
max_leaf_nodes = [40,50,60,70]
min_samples_leaf = [30,35,40,50,60,75,100,125,150,175]

dt_tuning_df = pd.DataFrame(columns = ["class_weight","max_leaf_nodes","min_samples_leaf","f1_train","f1_val",
                                      "roc_auc_train", "roc_auc_val","pr_auc_train", "pr_auc_val",
                                       "precision_c0","recall_c0",
                                       "precision_c1","recall_c1",
                                      "Tutar_Karşılıksız","Tutar_Onay"])


for i in range(400):
    crt = np.random.choice(criterion)
    cw = np.random.choice(class_weight)
    mln = np.random.choice(max_leaf_nodes)
    msl = np.random.choice(min_samples_leaf)
    
    tuned_tree = DecisionTreeClassifier(class_weight = cw, max_leaf_nodes = mln, min_samples_leaf = msl,
                                                     random_state = 0, max_depth = 7)
    
    tuned_tree.fit(X_train[all_features],y_train)
    tuned_pred_train = tuned_tree.predict(X_train[all_features])
    tuned_pred_val = tuned_tree.predict(X_val[all_features])
    
    
    roc_auc_train = roc_auc_score(y_train,tuned_pred_train)
    roc_auc_val = roc_auc_score(y_val,tuned_pred_val)
    
    pre, rec, _ = precision_recall_curve(y_train,tuned_pred_train)
    pr_auc_train = auc(pre, rec)    
    
    pre, rec, _ = precision_recall_curve(y_val,tuned_pred_val)
    pr_auc_val = auc(pre, rec)
    
    f1_train = f1_score(y_train,tuned_pred_train)
    f1_val = f1_score(y_val,tuned_pred_val)
    precision_c0 = precision_score(y_val,tuned_pred_val,pos_label = 0)
    precision_c1 = precision_score(y_val,tuned_pred_val,pos_label = 1)
    recall_c0 = recall_score(y_val,tuned_pred_val,pos_label = 0)
    recall_c1 = recall_score(y_val,tuned_pred_val,pos_label = 1)
    

    dt_tuning_df = dt_tuning_df.append({
        "class_weight" : cw,
        "max_leaf_nodes" : mln,
        "min_samples_leaf" : msl,
        "roc_auc_train" : roc_auc_train,
        "roc_auc_val" : roc_auc_val,
        "pr_auc_train" : pr_auc_train,
        "pr_auc_val" : pr_auc_val,
        "f1_train" : f1_train,
        "f1_val" : f1_val,
        "precision_c0" : precision_c0,
        "recall_c0" : recall_c0,
        "precision_c1" : precision_c1,
        "recall_c1" : recall_c1,
        "Tutar_Karşılıksız": tutar_karşılıksız,
        "Tutar_Onay": tutar_onay
        
    },ignore_index = True)
    
    if(i%100==0):
        print("Finished: ", i)
    


# ## Random Forest

# In[ ]:


n_estimators = [50,75,100,125]
class_weight = ["balanced","balanced_subsample",{0:1,1:5},{0:1,1:6},{0:1,1:7},{0:1,1:8},{0:1,1:9}]
max_leaf_nodes = [25,50,75,100]
min_samples_leaf = [10,15,20,25,30,35,40]
max_samples = [0.4,0.6,0.8,None]
max_features = [0.2,0.4,0.6]

rf_tuning_df = pd.DataFrame(columns = ["n_estimators",
                                       "class_weight",
                                       "max_leaf_nodes",
                                       "min_samples_leaf",
                                       "max_samples",
                                       "max_features",
                                       "f1_train",
                                       "f1_val",
                                      "roc_auc_train", "roc_auc_val","pr_auc_train", "pr_auc_val",
                                       "precision_c0","recall_c0",
                                       "precision_c1","recall_c1",
                                      "Tutar_Karşılıksız","Tutar_Onay"])

for i in range(500):
    est = np.random.choice(n_estimators)
    cw = np.random.choice(class_weight)
    msl = np.random.choice(min_samples_leaf)
    mln = np.random.choice(max_leaf_nodes)
    ms = np.random.choice(max_samples)
    mf = np.random.choice(max_features)
    
    tuned_tree = RandomForestClassifier(n_estimators = est, max_samples = ms,
                                        class_weight = cw,
                                        max_leaf_nodes = mln, 
                                        max_features = mf,
                                        max_depth = 10, min_samples_leaf = msl, 
                                                     random_state = 0)

    tuned_tree.fit(X_train[all_features],y_train)
    tuned_pred_train = tuned_tree.predict(X_train[all_features])
    tuned_pred_val = tuned_tree.predict(X_val[all_features])
    
    
    roc_auc_train = roc_auc_score(y_train,tuned_pred_train)
    roc_auc_val = roc_auc_score(y_val,tuned_pred_val)
    
    pre, rec, _ = precision_recall_curve(y_train,tuned_pred_train)
    pr_auc_train = auc(pre, rec)    
    
    pre, rec, _ = precision_recall_curve(y_val,tuned_pred_val)
    pr_auc_val = auc(pre, rec)
    
    f1_train = f1_score(y_train,tuned_pred_train)
    f1_val = f1_score(y_val,tuned_pred_val)
    precision_c0 = precision_score(y_val,tuned_pred_val,pos_label = 0)
    precision_c1 = precision_score(y_val,tuned_pred_val,pos_label = 1)
    recall_c0 = recall_score(y_val,tuned_pred_val,pos_label = 0)
    recall_c1 = recall_score(y_val,tuned_pred_val,pos_label = 1)
    

    rf_tuning_df = rf_tuning_df.append({
        "n_estimators": est,
        "class_weight": cw,
        "min_samples_leaf":msl,
        "max_leaf_nodes" : mln,
        "max_samples": ms,
        "max_features": mf,
        "roc_auc_train" : roc_auc_train,
        "roc_auc_val" : roc_auc_val,
        "pr_auc_train" : pr_auc_train,
        "pr_auc_val" : pr_auc_val,
        "f1_train" : f1_train,
        "f1_val" : f1_val,
        "precision_c0" : precision_c0,
        "recall_c0" : recall_c0,
        "precision_c1" : precision_c1,
        "recall_c1" : recall_c1,
        "Tutar_Karşılıksız": tutar_karşılıksız,
        "Tutar_Onay": tutar_onay
        
    },ignore_index = True)
    
    print(i)


# ## XGBoost

# In[ ]:


results_frame_XG = pd.DataFrame(columns=  ["learning_rate","n_estimators","scale_pos_weight",
                                      "max_depth","subsample","colsample_bytree","colsample_bylevel",
                                         "colsample_bynode","grow_policy","f1_train",
                                       "f1_val",
                                      "roc_auc_train", "roc_auc_val","pr_auc_train", "pr_auc_val",
                                       "precision_c0","recall_c0",
                                       "precision_c1","recall_c1",
                                      "Tutar_Karşılıksız","Tutar_Onay"])

learning_rate = [0.05,0.1,0.2,0.3,0.5]
n_estimators = [25,50,75,100,125,150]
scale_pos_weight = [3,5,6,7,8,9]
max_depth = [4,5,6,8,10]
subsample = [0.1,0.3,0.5,0.75,1]
colsample_bytree = [0.1,0.3,0.5,0.75,1]
colsample_bylevel = [0.1,0.3,0.5,0.75,1]
colsample_bynode = [0.1,0.3,0.5,0.75,1]
grow_policy = ["depthwise","lossguide"]


for i in range(2000):
    print(i)

    sel_learning_rate = np.random.choice(learning_rate)
    sel_n_estimators = np.random.choice(n_estimators)
    sel_scale_pos_weight = np.random.choice(scale_pos_weight)
    sel_max_depth = np.random.choice(max_depth)
    sel_subsample = np.random.choice(subsample)
    sel_colsample_bytree = np.random.choice(colsample_bytree)
    sel_colsample_bylevel = np.random.choice(colsample_bylevel)
    sel_colsample_bynode = np.random.choice(colsample_bynode)
    sel_grow_policy = np.random.choice(grow_policy)
    
    tuned_tree = XGBClassifier(objective = "binary:logistic", booster = "gbtree", random_state = 0,
                             max_depth = sel_max_depth,
                 scale_pos_weight = sel_scale_pos_weight,
                             learning_rate = sel_learning_rate
                             , n_estimators = sel_n_estimators
                          ,subsample = sel_subsample
                            ,colsample_bytree = sel_colsample_bytree
                            ,colsample_bylevel = sel_colsample_bylevel
                            ,colsample_bynode = sel_colsample_bynode
                            ,grow_policy = sel_grow_policy)
    
    tuned_tree.fit(X_train[all_features],y_train)
    tuned_pred_train = tuned_tree.predict(X_train[all_features])
    tuned_pred_val = tuned_tree.predict(X_val[all_features])
    
    
    roc_auc_train = roc_auc_score(y_train,tuned_pred_train)
    roc_auc_val = roc_auc_score(y_val,tuned_pred_val)
    
    pre, rec, _ = precision_recall_curve(y_train,tuned_pred_train)
    pr_auc_train = auc(pre, rec)    
    
    pre, rec, _ = precision_recall_curve(y_val,tuned_pred_val)
    pr_auc_val = auc(pre, rec)
    
    f1_train = f1_score(y_train,tuned_pred_train)
    f1_val = f1_score(y_val,tuned_pred_val)
    precision_c0 = precision_score(y_val,tuned_pred_val,pos_label = 0)
    precision_c1 = precision_score(y_val,tuned_pred_val,pos_label = 1)
    recall_c0 = recall_score(y_val,tuned_pred_val,pos_label = 0)
    recall_c1 = recall_score(y_val,tuned_pred_val,pos_label = 1)
    
    
    list_of_results =  { 
                        "learning_rate":sel_learning_rate,
                        "n_estimators":sel_n_estimators,
                        "scale_pos_weight":sel_scale_pos_weight,
                        "max_depth":sel_max_depth,
                        "subsample":sel_subsample,
                        "colsample_bytree":sel_colsample_bytree,
                        "colsample_bylevel":sel_colsample_bylevel,
                        "colsample_bynode":sel_colsample_bynode,
                        "grow_policy":sel_grow_policy,
                        "roc_auc_train" : roc_auc_train,
                        "roc_auc_val" : roc_auc_val,
                        "pr_auc_train" : pr_auc_train,
                        "pr_auc_val" : pr_auc_val,        
                        "f1_train" : f1_train,
                        "f1_val" : f1_val,
                        "precision_c0" : precision_c0,
                        "recall_c0" : recall_c0,
                        "precision_c1" : precision_c1,
                        "recall_c1" : recall_c1,
                        "Tutar_Karşılıksız": tutar_karşılıksız,
                        "Tutar_Onay": tutar_onay
                       }
        
    results_frame_XG = results_frame_XG.append(list_of_results, ignore_index = True)

