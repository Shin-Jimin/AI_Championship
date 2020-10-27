#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')


# # Read data

# In[2]:


f_name1 = '../RawData/AI챌린지 제출용 DB(ENT검사 진단명 메모 201009 ent)_train.xlsx'
df1 = pd.read_excel(f_name1)
df1.head()


# In[3]:


f_name2 = '../RawData/AI챌린지 제출용 DB(20191128고신대정상군 201009정리)_train.csv'
df2 = pd.read_csv(f_name2)
df2.head()


# # Concat DataFrame

# In[4]:


df1["Diagnosis_YN"] = np.full(len(df1), 1)
df2["Diagnosis_YN"] = np.full(len(df2), 0)

df1_cols = list(set(df1.columns.to_list())-set(df2.columns.to_list()))
df1 = df1.drop(df1_cols, axis=1)


# In[5]:


df = pd.concat([df1, df2], axis=0)


# In[6]:


df.info()


# # Train-Test split

# In[7]:


from sklearn.model_selection import train_test_split

y = df["Diagnosis_YN"]
X = df.drop(["Diagnosis_YN"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=215)


# In[8]:


# 스코어보드 제출용 정답 파일
y_test.to_csv('../Data/Test_y.csv')


# In[9]:


# 모델 학습과 평가를 위한 데이터 셋
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv('../Data/Train_all.csv', index=False)
test.to_csv('../Data/Test_all.csv', index=False)

