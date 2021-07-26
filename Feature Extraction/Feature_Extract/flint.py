#!/usr/bin/env python
# coding: utf-8

# **②  基于flint.csv 中的特征：** 位置[83,104] + [127,144] +[253,261]+[271,297]
# 1. <font color=green>**count_domain_ip**</font> [97,100]：一天内映射的不同ip地址的个数 ，对数据取sum\var\mean\std
# 2. <font color=green>**count_domain_flintType**</font> count_domain_flintType[93,96]：一天内映射的不同类型的个数  ，对数据取sum\var\mean\std
# 3. <font color=green>**sub_domain_requestCnt**</font>sub_domain_requestCnt[89,92]: 一天内requestCnt的最大值和最小值的差值，对数据取sum\var\mean\std
# 4. count_domain_requestCnt[101,104]：一天内不同requestCnt的个数，对数据取sum\var\mean\std
# 5. <font color=green>**count_ptr**</font>[83]: 域名映射的ip地址和其他域名共享的不同的域名个数：  
# E.g: 域名a映射的三个ip地址分别被其他不同的3、2、1个域名映射，则count_ptr=6
# 6. <font color=red>**jaccard_rr_no**</font>[253,261]: 计算域名解析记录与各个恶意域家族解析记录的jaccard系数
#      举例：
#         域A的解析记录值(encoded_value)列表【IP-1，cname-1，IPv6-1】
#         域B的解析记录值(encoded_value)列表【IP-1，cname-1，IPv6-4】
#         Jaccard系数：相同交集IP-1，cname-1=2个，并集IP-1，cname-1，IPv6-1，IPv6-4=4个，则得到相似系数 2/4 =0.5
#         实际使用统计与带标签的良性/恶意域的解析记录集，计算待预测域与带标签的良性/恶意域的解析记录列表的jaccard系数
# 7. IP关联—— <font color=red>**ip_no_counts**</font> [136,144]
# - ip_0_counts：该域名映射过的ip列表[a1,a2,...,an],ai曾被ki个属于family_no=0的恶意域映射过，则【ip_0_counts】=Σn(ki)
# - ...
# - 分析：统计了每个encoded_value(A/AAAA记录)的重要性，使用加权形式表示
# 
# 8. IP关联——<font color=red>**ip_no_dif_counts**</font>[127,135]
# - ip_1_counts:该域名映射的ip曾被【ip_1_counts】个属于family_no=1的恶意域映射过
# - ...
# - 分析：直接计算关联个数
# -------------------------------------------
# 9. 补充：(原来部分直接等价于***_sum)
#  - ' <font color=green>**sum_sub_domain_requestCnt**</font>',- 84
#  - 'sum_count_domain_flintType', -85
#  - ' <font color=green>**sum_count_domain_ip**</font>', -86
#  - 'sum_count_domain_requestCnt' -87
#  - '<font color=green>**sum_domain_ip**</font>sum_domain_ip' -88
# 10. 补充：(原来部分直接置0)
#     -    [271,297]
#     -   jaccard_cname_no
#     -   jaccard_ns_no
#     -   jaccard_mx_no
#     -   jaccard_ip_no
# 11. 新增：
#     - byday: sum_request_day_(多个描述)
#     - overall:
#         - ip_mali_counts
#         - ip_mali_dif_counts
#         - jaccard_rr_mali
#         - jaccard_cname_mali
#         - jaccard_ns_mali
#         - jaccard_mx_mali
#         - jaccard_ip_mali

# In[1]:


import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import re
import os



class FLINT:
    def __init__(self,path_train_flint=r'Data/flint.csv',path_train_fqdn=r'Data/fqdn.csv'):

        train_flint = pd.read_csv(path_train_flint)
        train_fqdn = pd.read_csv(path_train_fqdn)



        # In[5]:


        if train_flint.columns.to_list()[0]=='fqdn_no':
            train_flint.rename(columns={'fqdn_no':'fqdn_no_x'}, inplace = True)



        # ## 固定代码部分

        # In[6]:
        day_list =list(train_flint.date.value_counts().reset_index(name = 'counts')['index'])

        day_len = len(day_list)
        day_dic = dict(zip(day_list, range(day_len)))
        def make_attrib(str1,df,day_len):

            for i in range(day_len):
                df[str1+str(i)]=0

        print("Extracting FLINT Feature:")
        if (os.path.isfile(r'Feature_Extract\tmp_flint.csv')):
            final_flint=pd.read_csv(r'Feature_Extract\tmp_flint.csv')
        else:

            final_flint = pd.DataFrame()

            final_flint['fqdn_no'] =train_fqdn ['fqdn_no']

            make_attrib('count_domain_ip_',final_flint,day_len)
            make_attrib('count_domain_flintType_',final_flint,day_len)
            make_attrib('sum_request_day_',final_flint,day_len)
            make_attrib('sub_domain_requestCnt_',final_flint,day_len)
            make_attrib('count_domain_requestCnt_',final_flint,day_len)


            for i in tqdm(range(day_len)):
                df = train_flint[train_flint['date']==day_list[i]]
                set_day = list(df.drop_duplicates(['fqdn_no_x'])['fqdn_no_x'])
                num = day_dic[day_list[i]] + 1
                for j in range(len(final_flint)):
                    if(final_flint.iloc[j,0] in set_day):
                        temp_df = df[df['fqdn_no_x']==final_flint.iloc[j,0]]
                        count_ip = len(temp_df[temp_df['flintType'].isin([1,28])].drop_duplicates(['encoded_value']))
                        count_flintType_day = len(temp_df.drop_duplicates(['flintType']))
                        sum_request_day = np.sum(temp_df['requestCnt'])
                        sub_domain_requestCnt = np.max(temp_df['requestCnt'])-np.min(temp_df['requestCnt'])
                        count_dif_requestCnt = len(temp_df ['requestCnt'].drop_duplicates())

                        final_flint.iloc[j,num]=count_ip
                        final_flint.iloc[j,num+day_len ]=count_flintType_day
                        final_flint.iloc[j,num+day_len *2] = sum_request_day
                        final_flint.iloc[j,num+day_len *3] =  sub_domain_requestCnt
                        final_flint.iloc[j,num+day_len *4] =  count_dif_requestCnt
            final_flint.to_csv(r'Feature_Extract\tmp_flint.csv',index=False)


        # In[8]:


        def make_attrib_v2(str_list,add_list,df):
            for str_attr in str_list:
                for add_attr in add_list:
                    df[str_attr+add_attr] = 0

        len_df1 = final_flint.shape[1]


        str_list = ['count_domain_ip_','count_domain_flintType_','sum_request_day_','sub_domain_requestCnt_','count_domain_requestCnt_']
        add_list = ['sum','var','std','mean','max','min','percent25','median','percent75']
        make_attrib_v2(str_list ,add_list ,final_flint)


        final_flint['sum_ip']=0


        # In[ ]:


        add_attr_len = len(add_list)
        str_list_len = len(str_list)
        if (os.path.isfile(r'Feature_Extract\tmp_flint2.csv')):
            flint_feature = pd.read_csv(r'Feature_Extract\tmp_flint2.csv')
        else:
            for i in tqdm(range(len(final_flint))):
                temp_df = train_flint[train_flint['fqdn_no_x']==final_flint.iloc[i,0]]
                sum_ip  = len(temp_df[temp_df['flintType'].isin([1,28])].drop_duplicates(['encoded_value']))
                sum_request = np.sum(temp_df['requestCnt'])
                for j in range(str_list_len):

                    tmp_1 = list(final_flint.iloc[i,1+day_len*j:1+day_len*(j+1)])

                    v1 = np.array(tmp_1)
                    v1= v1[~np.isnan(v1)]
                    v1 = list(v1)
                    if j==1 or j==3:
                        v1.append(0)
                    if(len(v1)>1):
                        final_flint.iloc[i,len_df1+j*add_attr_len]=np.sum(v1)
                        final_flint.iloc[i,len_df1+j*add_attr_len+1]=np.var(v1)
                        final_flint.iloc[i,len_df1+j*add_attr_len+2]=np.std(v1)
                        final_flint.iloc[i,len_df1+j*add_attr_len+3]=np.mean(v1)
                        final_flint.iloc[i,len_df1+j*add_attr_len+4]=np.max(v1)
                        final_flint.iloc[i,len_df1+j*add_attr_len+5]=np.min(v1)
                        tmp_3 = np.percentile(v1, [25, 50, 75])
                        final_flint.iloc[i,len_df1+j*add_attr_len+6]=int(tmp_3 [0])
                        final_flint.iloc[i,len_df1+j*add_attr_len+7]=int(tmp_3 [1])
                        final_flint.iloc[i,len_df1+j*add_attr_len+8]=int(tmp_3 [2])
                final_flint.iloc[i,len_df1+str_list_len*add_attr_len] = sum_ip


            flint_attr = final_flint.columns.values.tolist()



            flint_feature = final_flint.drop(flint_attr[1:len_df1],axis=1)


        # # In[ ]:
        #
        #
        #
        flint_feature['count_ptr'] = 0
        col_num = flint_feature.shape[1]


        print("continue Extract")
        tmp_df_fqdn = train_flint.loc[ train_flint['flintType'].isin([1,28])].drop_duplicates(['fqdn_no_x','encoded_value'])[['fqdn_no_x','encoded_value']]
        for i in tqdm(range(len(flint_feature))):
            tmp_df= tmp_df_fqdn[tmp_df_fqdn['fqdn_no_x']==train_fqdn.iloc[i,1]]
            tmp_n_df = tmp_df_fqdn[tmp_df_fqdn['fqdn_no_x']!=train_fqdn.iloc[i,1]]
            df_tmp = len(pd.merge(tmp_df,tmp_n_df,on='encoded_value'))
            flint_feature.iloc[i,col_num-1] = df_tmp


        # In[13]:


        flint_feature['sum_sub_domain_requestCnt'] =0
        flint_feature['sum_count_domain_flintType']=0
        flint_feature['sum_count_domain_ip'] = 0
        flint_feature['sum_count_domain_requestCnt']=0



        for i in tqdm(range(len(flint_feature))):
            tmp_df= train_flint[train_flint['fqdn_no_x']==train_fqdn.iloc[i,1]]

            if len(tmp_df)>0:
                flint_feature.loc[i,'sum_sub_domain_requestCnt'] = np.max(tmp_df['requestCnt'].to_list())-np.min(tmp_df['requestCnt'].to_list())
                flint_feature.loc[i,'sum_count_domain_flintType'] = len(np.unique(tmp_df['flintType'].to_list()))
                flint_feature.loc[i,'sum_count_domain_ip'] = len(tmp_df[tmp_df['flintType'].isin([1,28])].drop_duplicates(['encoded_value']))
                flint_feature.loc[i,'sum_count_domain_requestCnt'] = len(tmp_df.drop_duplicates(['requestCnt'])['requestCnt'].to_list())


        flint_feature.to_csv(r'Feature_Extract/02_Feature_SRC/flint_feature.csv',index=False)






