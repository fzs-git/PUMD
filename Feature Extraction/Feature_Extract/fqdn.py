#!/usr/bin/env python
# coding: utf-8

# **① 基于fqdn.csv中encoded_fqdn 的特征：**  位置[73,82]  
# 例： 0aa0a00a000aaa000aaaaa0a0a00a000.aaa.[aaaaa].[aaaaaa][aaaa].com
# 1. <font color=green>**len_domain**</font>:域名的长度.     
# 2. <font color=green>**num_point**</font>:域名中子域的个数   —>4(小数点的个数)
# 3. <font color=green>**num_special**</font>: 特殊字符的个数（例如”-”） —>0
# 4. <font color=green>**count_a_d**</font>:字母和数字的转换频率     
# 5. num_a:字母的个数      —>(不算[]里的)
# 6. <font color=green>**num_l**</font>:数字的个数      
# 7. len_l:连续数字字符的长度      —>3
# 8. len_max_word:词语的最长长度       —>6
# 9. <font color=green>**count_word**</font>:一个域名中包含词语的个数    —>3
# 10. <font color=green>**count_dif_word**</font>:一个域名中有几种不同的词语的长度   —>3

# In[1]:


import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import re


# In[2]:
class FQDN:
    def __init__(self,dir_fqdn=r'Data/fqdn.csv'):



        train_fqdn = pd.read_csv(dir_fqdn)


        # In[3]:


        train_fqdn['len_domain']=0
        train_fqdn['num_point']=0
        train_fqdn['num_special']=0
        train_fqdn['num_a']=0
        train_fqdn['num_l']=0
        train_fqdn['count_word']=0
        train_fqdn['count_a_d']=0
        train_fqdn['len_max_word']=0
        train_fqdn['count_dif_word']=0
        train_fqdn['len_l']=0

        print("Extracting FQDN Feature:")
        for i in tqdm(range(len(train_fqdn))):
            train_fqdn.iloc[i,2] = len(train_fqdn.iloc[i,0].replace('[','').replace(']',''))#len_domain
            train_fqdn.iloc[i,3] = train_fqdn.iloc[i,0].count('.')#num_point
            train_fqdn.iloc[i,4] = len((''.join(train_fqdn.iloc[i,0].split('.')[:-1])).replace('[','').replace(']','').replace('a','').replace('0',''))#num_special
            train_fqdn.iloc[i,5] = (''.join(train_fqdn.iloc[i,0].split('.')[:-1])).count('a')#num_a
            train_fqdn.iloc[i,6] = (''.join(train_fqdn.iloc[i,0].split('.')[:-1])).count('0')#num_l
            train_fqdn.iloc[i,7] = train_fqdn.iloc[i,0].count('[')#count_word
            train_fqdn.iloc[i,8] = train_fqdn.iloc[i,0].count('a0')+train_fqdn.iloc[i,0].count('0a')#count_a_d
            lst=re.findall(r'\[(.+?)\]', train_fqdn.iloc[i,0])
            if(len(lst)==0):
                train_fqdn.iloc[i,9] = 0
                train_fqdn.iloc[i,10] = 0
            else:
                train_fqdn.iloc[i,9] = len(max(lst,key = lambda x:len(x)))#len_max_word
                train_fqdn.iloc[i,10] = len(set(lst)) #'count_dif_word'
            lst1 = re.findall(r'0*', train_fqdn.iloc[i,0])
            if(len(lst1)==0):
                train_fqdn.iloc[i,11] = 0
            else:
                train_fqdn.iloc[i,11] = len(max(lst1,key = lambda x:len(x)))


        train_fqdn.to_csv(r'Feature_Extract/02_Feature_SRC/fqdn_feature.csv',index=False)





