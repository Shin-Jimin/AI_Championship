#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')


# # Read data

# In[2]:


f_name1 = '../Result/predict_D_xgb_batch_included.csv' #ABACUS 'batch prediction' 결과 파일
df1 = pd.read_csv(f_name1)
df1


# In[3]:


df1["0_bst_xgb_guess"].value_counts()


# # Evaluate prediction result

# In[4]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score


# In[5]:


# 최적 스코어를 확률로 환산하는 함수
def convert_score_to_prob(score):
    x = np.log(2)/40*(score-600)
    prob = np.exp(x) / (1+np.exp(x))
    
    return prob


# In[6]:


cutoff_prob = convert_score_to_prob(572)


# In[7]:


y_true = df1["Diagnosis_YN"]
y_pred = (df1["0_bst_xgb_score"]>cutoff_prob)*1

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(f1_score(y_true, y_pred))


# # Save predict_y

# In[8]:


# 스코어보드 제출용 파일
y_pred.to_csv("../Result/201022_xgb_predict_y.csv")

