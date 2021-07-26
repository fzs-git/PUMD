#!/usr/bin/env python
# coding: utf-8

# ## 特征

# 1. Validity period of domain     过期时间--注册时间 expiresdate-createddate
# 2. Number of distinct NS        ns的个数 nameservers
# 3. Similarity of NS domain name   ns名称的相似性

# 1. Validity period of domain （valid_time） 过期时间--注册时间 expiresdate-createddate
# 2. Number of distinct NS   (NS_num)     ns的个数 nameservers
# 3. Similarity of NS domain name  (simi_ns) ns名称的相似性
# 
# 4. whois_num                whois记录数量
# 5. is_date_exist             是否createddate、expiresdate、updateddate 都存在
# 6. info_none_num             统计whois信息中条目为none的个数
# 7. max_len_admin_email         admin_email的最大长度

# In[1]:


import pandas as pd
import json
import csv
import numpy as np
import os
from tqdm import tqdm
import Levenshtein


# In[2]:

class WHOIS:
    def __init__(self,whois_path=r'Data/whois.json',path_train_fqdn=r'Data/fqdn.csv',):
        with open(whois_path, 'r') as f:
            temp = json.loads(f.read())
            #temp数组内有n个whois信息条目,如下所示，对于fqdn_no有重复条目出现
            '''
            {'fqdn_no': 'fqdn_0',
             'createddate': 777182400000,
             'expiresdate': 1534478400000,
             'updateddate': 1500196908000,
             'nameservers': ['ns1.msft.net',
              'ns1a.o365filtering.com',
              'ns2.msft.net',
              'ns2a.o365filtering.com',
              'ns3.msft.net',
              'ns4.msft.net',
              'ns4a.o365filtering.com'],
             'admin_email': None,
             'registrant_country': None,
             'registrant_email': None,
             'registrant_state': None,
             'tech_email': None,
             'r_whoisserver_list': ['whois.markmonitor.com'],
             'whoisserver': 'whois.markmonitor.com',
             'sponsoring': None}
            '''

        df = pd.read_csv(path_train_fqdn)
        fqdn_list = list(df['fqdn_no'])

        save_path = r'Feature_Extract\json_fqdn_list.npy'
        print("Extracting WHOIS Feature:")
        json_fqdn_list = []
        if os.path.isfile(save_path):
            #取
            a=np.load(save_path,allow_pickle=True)
            json_fqdn_list=a.tolist()
        else:
        #存
            #对于待检测域名样本提取相关的whois信息列表json_fqdn_list

            for j in range(len(fqdn_list)):
                j_list = []
                for i in range(len(temp)):
                    if(temp[i]['fqdn_no'] == fqdn_list[j]):
                        j_list.append(temp[i])
                json_fqdn_list.append(j_list)
            a = np.array(json_fqdn_list)
            np.save(save_path,a)
        #NS相似性代码

        def Similarity_function(list_ns):
            dis = 0
            for i in list_ns:
                for j in list_ns:
                    dis += Levenshtein.distance(i,j)
            count = dis/len(list_ns)
            return count


        dict_list = list(json_fqdn_list[0][1])
        fqdn_feature_list = []
        for i in tqdm(range(len(json_fqdn_list))):
            feature = []
            tmp1 = []
            tmp2 = []
            tmp3 = []
            tmp5= []
            tmp6 = []
            tmp7 = []
            for j in range(len(json_fqdn_list[i]) ):

                    value1 = 0
                    if(json_fqdn_list[i][j]['expiresdate'] is not None):
                         if(json_fqdn_list[i][j]['createddate'] is not None):
                                #Validity period of domain （valid_time）
                                value1 = json_fqdn_list[i][j]['expiresdate'] - json_fqdn_list[i][j]['createddate']

                    value2 = 0
                    value3 = 0
                    if(json_fqdn_list[i][j]['nameservers'] is not None):
                            #Number of distinct NS (NS_num) ns的个数 nameservers
                            value2 =  len(json_fqdn_list[i][j]['nameservers'])
                            #Similarity of NS domain name (simi_ns) ns名称的相似性
                            if(value2>1):
                                value3 = Similarity_function(json_fqdn_list[i][j]['nameservers'])
                    #is_date_exist 是否createddate、expiresdate、updateddate 都存在
                    value5 = 1
                    for m in dict_list[1:4]:
                        if (json_fqdn_list[i][j][m] is None):
                             value5 = 0
                    #info_none_num 统计whois信息中条目为none的个数
                    value6 = 0
                    for m in dict_list:
                        if (json_fqdn_list[i][j][m] is None):
                            value6 +=1
                    #max_len_admin_email admin_email的最大长度
                    value7 = 0
                    if(json_fqdn_list[i][j]['admin_email'] is not None):
                        value7 = len(json_fqdn_list[i][j]['admin_email'][0])
                    tmp1.append(value1)
                    tmp2.append(value2)
                    tmp3.append(value3)
                    tmp5.append(value5)
                    tmp6.append(value6)
                    tmp7.append(value7)

            #whois_num whois记录数量v
            v4 = len(json_fqdn_list[i])
            v1 = np.mean(tmp1)
            v2 = np.mean(tmp2)
            v3 = np.mean(tmp3)
            v5 = np.mean(tmp5)
            v6 = np.mean(tmp6)
            v7 = np.max(tmp7)
            tmp = [v1,v2,v3,v4,v5,v6,v7]
            fqdn_feature_list.append(tmp)


        # In[8]:


        colum0 =['fqdn_no','valid_time','NS_num','simi_ns','whois_num','is_date_exist','info_none_num','max_len_admin_email']
        df_feature = pd.DataFrame(columns= colum0 )


        # In[9]:


        for i in range(len(fqdn_feature_list)):
            a = []
            a.append('fqdn_'+str(i))
            a.extend(fqdn_feature_list[i])

            df_feature.loc[len(df_feature)] = a
        df_feature.to_csv(r'Feature_Extract/02_Feature_SRC/whois_feature.csv',index=False)




