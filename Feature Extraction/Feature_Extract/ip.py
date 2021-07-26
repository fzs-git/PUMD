#!/usr/bin/env python
# coding: utf-8

# **④ 基于ip.csv / ipv6.csv中的特征：**
# 1. country_count：统计一个域映射的IP所属国家数 [145]
# 2. subvision_count：统计一个域映射的IP所属地区数[146]
# 3. subvision_jaccard_no: 统计一个域映射的IP所属地区与恶意域的IP所属地区相似性[147,155]
# - 举例：
#     域A映射的IP所属地区【‘北京’，‘上海’，‘天津’】
#     域A映射的IP所属地区【‘北京’，‘上海’，‘湖南’】
#     Jaccard系数：相同交集北京’，‘上海’=2个，并集‘北京’，‘上海’，‘天津’，‘湖南’=4个，则得到相似系数 2/4 =0.5
#     实际使用统计与带标签的良性/恶意域的IP所属区域集，计算待预测域与带标签的良性/恶意域的IP所属区域列表的jaccard系数

# In[1]:


import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import re
class IP:
    def __init__(self,path_df_label=r'Feature_Extract/Label_For_Feature_Extract/label.csv',
                 path_train_fqdn=r'Data/fqdn.csv',path_train_flint = r'Data/flint.csv',
                 path_train_access=r'Data/access.csv',path_train_ip=r'Data/ip.csv',path_train_ipv6=r'Data/ipv6.csv'):

        T4_train_ip = pd.read_csv(path_train_ip)
        T4_train_ipv6 = pd.read_csv(path_train_ipv6)
        T4_train_fqdn = pd.read_csv(path_train_fqdn)
        T4_train_flint = pd.read_csv(path_train_flint)
        T4_train_label = pd.read_csv(path_df_label)


        def jaccard_count(list1,list2):
            m = len(set(list1).union(set(list2)))
            n = len(set(list1).intersection(set(list2)))
            return (n/m)



        T4_train_flint.rename(columns={'fqdn_no_x':'fqdn_no'},inplace=True)



        # In[7]:


        train_ip = T4_train_ip.loc[:,['encoded_ip','country','subdivision']]
        train_ipv6 = T4_train_ipv6.loc[:,['encoded_ip','country','subdivision']]
        train_ip = pd.concat([train_ipv6,train_ip],axis=0)
        train_flint = T4_train_flint.loc[:,['fqdn_no','encoded_value']].drop_duplicates()
        train_df_ip = pd.merge(train_flint,train_ip,left_on='encoded_value',right_on = 'encoded_ip')


        # In[8]:


        final_ip = pd.DataFrame()
        final_ip['fqdn_no'] = T4_train_fqdn['fqdn_no']
        final_ip['country_count'] = 0
        final_ip['subvision_count'] = 0
        # final_ip['subvision_jaccard_mali'] = 0
        mali_subvision_list = np.unique(train_df_ip[train_df_ip['fqdn_no'].isin(T4_train_label['fqdn_no'].to_list())]['subdivision'].to_list())
        for i in tqdm(range(len(final_ip))):
            tmp_df = train_df_ip[train_df_ip['fqdn_no'] == final_ip.iloc[i,0]]
            final_ip.loc[i,'country_count'] =len( np.unique(tmp_df['country'].to_list()))
            final_ip.loc[i,'subvision_count'] =len( np.unique(tmp_df['subdivision'].to_list()))
            # final_ip.loc[i,'subvision_jaccard_mali']= jaccard_count(np.unique(tmp_df['subdivision'].to_list()),mali_subvision_list)
        final_ip.to_csv(r'Feature_Extract/02_Feature_SRC/ip.csv',index=False)

