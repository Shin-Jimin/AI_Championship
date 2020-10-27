#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import pandas as pd


# In[ ]:


#채점에 사용할 함수 정의
from sklearn.metrics import f1_score


# In[ ]:


# 정답(label_data) 및 제출 파일(predict_data) 형식에 맞게 불러오기
predict_data = pd.read_csv(sys.argv[2], index_col=0)
label_data = pd.read_csv(sys.argv[1], index_col=0)

score = f1_score(label_data, predict_data)
print("score: %.4f", %score)


# In[ ]:




