#!/usr/bin/env python
# coding: utf-8

# [1,72]+[105,126]+[226,270]
# - byday:
#     - sum_day_request:一天中一共被请求过多少次（求一天中count的和）[262,270]
#     - sum_day_ip:一天中一共被多少个encoded_ip请求过 **有区别** [119,122]
#     - count_hour_request：一天中有几个小时在访问该域名【按小时聚合（原按10分钟聚合）】**原/新增**  [111,114]
#     - max_request：一天中各小时请求该域名的最大请求数【按小时聚合（原按10分钟聚合）】**原/新增**  [115,118]
#     - sub_ip：一天中各小时最大请求数max减最小请求数min的差值(max_request里的做法) 【按小时聚合（原按10分钟聚合）】**原/新增**  [123,126]
#     - req_count_percent:以小时为粒度统计最频繁请求域的时间，统计该时间段内请求量占比全天请求量 [1,9]
#     - req_time_no_:统计请求时间倾向性：将全天切成时段（6），统计各时段内访问量 [10,72]
# - overall:
#     - count_ip:访问总次数 np.sum(temp_df['request_cnt'].to_list())
#     - count_diff_ip：一个域名具有的不同encoded_ip值的个数 **有区别** [107]
#     - count_diff_request：一个域名具有的不同sum(count)值的个数[108]
#     - count_loss：有几天没有被请求
#     - (sum_months_request  ==  sum_day_request_sum)[109]
# - 新增：
#     - byday:
#         - count_hour_request **原/新增**  
#         - max_request **原/新增**  
#         - sub_ip **原/新增**  
#         - count_day_dif 多少个不同的小时访问值 **新增**
#         - count_hour_loss：多少个小时未被访问**新增**
# - 缺少：
#     - overall
#         - sub_request: 一个域名的所有小时的最大sum(count)减最小sum(count)的差值[106]
# 
#  -------------------------------------------------------------
#      - 不能绑定data5的IP和恶意关系因此未编写
# 
#         - 12. ip_rec_domain：每个客户端请求DNS域名解析的统计信息 [226,234]
#             - 举例：IP1，IP2，IP3均访问域A，分别统计IP1/2/3访问域Ａ总次数IP1_rec_count,IP2_rec_count,IP3_rec_count，填充列表［IP1_rec_count, IP2_rec_count,IP3_rec_count］
#             - 对列表计算总数、算数平均、中位数、标准差、最小值、最大值、上下四分位数统计值
#         - 13. ip_related_list：域关联的IP列表，计算相应的统计量，包括算术平均值，中位数，相应的标准差，最小值，最大值，下四分位数和上四分位数
#             - 举例：域A部署在其上的IP的数量为3。通过第三个特征表示所部署的IP的特性。如果IP1上有100个域，IP2上有20个域，IP3上有1000个域，则域A的相同IP上的关联域列表为[100，20，1000]。计算列表的统计量 [235,243]
#         - 14. <font color=red>**jaccard_ip_no**</font>：计算域的请求集IP与各恶意域家族的Jaccard相似系数 [244,261]
#             - 举例：
#                 查询域A的IP列表【IP1，IP2，IP3】
#                 查询域B的IP列表【IP2，IP3，IP4】
#                 Jaccard系数：相同交集IP2，IP3=2个，并集IP1/2/3/4=4个，则得到相似系数 2/4 =0.5
#                 实际使用统计各个恶意家族的请求集IP，计算待预测域与各恶意家族IP列表的jaccard系数
# 

# ## 》access.csv
# [5,4,2,]
# - sum_day_count[sum_day_request-5]: 每天的请求次数，求91天的sum/var/std/mean/max/min/percent25/median/percent75
# - sum_day_ip[sum_day_ip-4]：每天请求的IP数，求91天的sum/var/std/mean/max/min/percent25/median/percent75
# - count_hour_req[count_hour_request-2]：每天被请求的小时个数，求91天的sum/var/std/mean/max/min/percent25/median/percent75
# - sub_day_count：每天请求次数最大值和最小值的差值，求91天的sum/var/std/mean/max/min/percent25/median/percent75
# - count_day_dif：每天不同请求值的个数，求91天的sum/var/std/mean/max/min/percent25/median/percent75
# - count_dif_ip[count_diff_ip-8]：全天数请求的IP个数
# - count_dif_count[count_diff_request-9]：全天数请求值的个数
# - count_day_loss[count_loss-11]：有几天未被请求
# - count_hour_loss：有几个小时未被请求
# - ip_req_domain：请求域名的IP地址和其他域名共享的不同IP地址个数
# - - E.g： 请求域名a的三个ip地址分别请求不同的3、2、1个域名，则ip_req_domain=6

# In[1]:


import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import re
import os
# In[2]:
class ACCESS:
    def __init__(self,dir_fqdn=r'Data/fqdn.csv',dir_access=r'Data/access.csv'):

        T4_train_access =pd.read_csv(dir_access)
        T4_train_fqdn =pd.read_csv(dir_fqdn)



        # In[2]:


        def make_attrib(str1,df,day_len):

            for i in range(day_len):
                df[str1+str(i)]=0
        def make_attrib_v2(str_list,add_list,df):
            for str_attr in str_list:
                for add_attr in add_list:
                    df[str_attr+add_attr] = 0



        T4_train_access['date'] = T4_train_access['time'].floordiv(1000000, fill_value = 0)
        T4_train_access['hour'] = T4_train_access['time'].mod(1000000, fill_value = 0).floordiv(10000, fill_value = 0)
        train_access = T4_train_access
        train_fqdn = T4_train_fqdn


        # - 12. ip_rec_domain：每个客户端请求DNS域名解析的统计信息 [226,234]
        #             - 举例：IP1，IP2，IP3均访问域A，分别统计IP1/2/3访问域Ａ总次数IP1_rec_count,IP2_rec_count,IP3_rec_count，填充列表［IP1_rec_count, IP2_rec_count,IP3_rec_count］
        #             - 对列表计算总数、算数平均、中位数、标准差、最小值、最大值、上下四分位数统计值
        # - 13. ip_related_list：域关联的IP列表，计算相应的统计量，包括算术平均值，中位数，相应的标准差，最小值，最大值，下四分位数和上四分位数
        #             - 举例：域A部署在其上的IP的数量为3。通过第三个特征表示所部署的IP的特性。如果IP1上有100个域，IP2上有20个域，IP3上有1000个域，则域A的相同IP上的关联域列表为[100，20，1000]。计算列表的统计量 [235,243]

        day_list = list(train_access.date.value_counts().reset_index(name='counts')['index'])

        day_len = len(day_list)

        day_dic = dict(zip(day_list, range(day_len)))

        # In[6]:

        print("Extracting ACCESS Feature:")
        if (os.path.isfile(r'Feature_Extract\tmp_access.csv')):
            final_access=pd.read_csv(r'Feature_Extract\tmp_access.csv')
        else:
            final_access = pd.DataFrame()
            final_access ['fqdn_no'] =train_fqdn ['fqdn_no']





            make_attrib('sum_day_request_',final_access,day_len)#5
            make_attrib('sum_day_ip_',final_access,day_len)#4
            make_attrib('count_hour_request_',final_access,day_len)#2
            make_attrib('max_request_',final_access,day_len)#3
            make_attrib('sub_ip_',final_access,day_len)#6
            make_attrib('count_day_dif_',final_access,day_len)#新增,多少个不同的小时访问值

            make_attrib('req_count_percent_',final_access,day_len)
            make_attrib('req_time_0_',final_access,day_len)
            make_attrib('req_time_1_',final_access,day_len)
            make_attrib('req_time_2_',final_access,day_len)
            make_attrib('req_time_3_',final_access,day_len)
            make_attrib('req_time_4_',final_access,day_len)
            make_attrib('req_time_5_',final_access,day_len)

            for i in tqdm(range(day_len)):
                df = train_access[train_access['date']==day_list[i]]
                set_day = list(df.drop_duplicates(['fqdn_no'])['fqdn_no'])
                num = day_dic[day_list[i]] + 1
                for j in range(len(final_access)):
                    if(final_access.iloc[j,0] in set_day):
                        temp_df = df[df['fqdn_no']==final_access.iloc[j,0]]
                        sum_day_count = np.sum(temp_df['count']) #sum_day_count
                        sum_day_ip = len(temp_df.drop_duplicates(['encoded_ip'])['encoded_ip'].to_list())#sum_day_ip
                        #count_day_hour
                        count_day_hour = len(temp_df.drop_duplicates(['hour'])['hour'].to_list())
                        #按Hour聚合
                        tmp_hour_df = temp_df.loc[:,['count','hour','date']].groupby(['hour','date'])['count'].sum().reset_index(name='count_hour_req')
                        max_request_hour = np.max(tmp_hour_df['count_hour_req'])
                        sub_ip = np.max(tmp_hour_df['count_hour_req'])-np.min(tmp_hour_df['count_hour_req']) #sub_day_count
                        count_day_dif = len(tmp_hour_df['count_hour_req'].drop_duplicates())#count_day_dif

                        req_count_percent = max_request_hour/sum_day_count
                        req_time_0 = np.sum(temp_df[temp_df['hour'].isin([0,1,2,3])]['count'].to_list())
                        req_time_1 = np.sum(temp_df[temp_df['hour'].isin([4,5,6,7])]['count'].to_list())
                        req_time_2 = np.sum(temp_df[temp_df['hour'].isin([8,9,10,11])]['count'].to_list())
                        req_time_3 = np.sum(temp_df[temp_df['hour'].isin([12,13,14,15])]['count'].to_list())
                        req_time_4 = np.sum(temp_df[temp_df['hour'].isin([16,17,18,19])]['count'].to_list())
                        req_time_5 = np.sum(temp_df[temp_df['hour'].isin([20,21,22,23])]['count'].to_list())

                        final_access.iloc[j,num] = sum_day_count
                        final_access.iloc[j,num+day_len ] = sum_day_ip
                        final_access.iloc[j,num+day_len *2] = count_day_hour
                        final_access.iloc[j,num+day_len *3] = max_request_hour
                        final_access.iloc[j,num+day_len *4] =  sub_ip
                        final_access.iloc[j,num+day_len *5] =  count_day_dif
                        final_access.iloc[j,num+day_len *6] =  req_count_percent
                        final_access.iloc[j,num+day_len *7] = req_time_0
                        final_access.iloc[j,num+day_len *8] = req_time_1
                        final_access.iloc[j,num+day_len *9] = req_time_2
                        final_access.iloc[j,num+day_len *10] = req_time_3
                        final_access.iloc[j,num+day_len *11] = req_time_4
                        final_access.iloc[j,num+day_len *12] = req_time_5

            final_access.to_csv(r'Feature_Extract\tmp_access.csv',index=False)
        len_df1 = final_access.shape[1]
        str_list = ['sum_day_request_','sum_day_ip_','count_hour_request_','max_request_','sub_ip_','count_day_dif_',
                    'req_count_percent_','req_time_0_','req_time_1_','req_time_2_','req_time_3_','req_time_4_','req_time_5_',
                    'ip_rec_domain_','ip_related_list_']#新增2
        add_list = ['sum','var','std','mean','max','min','percent25','median','percent75']
        make_attrib_v2(str_list ,add_list ,final_access)




        final_access['count_ip'] = 0 #1
        final_access['count_diff_ip'] = 0 #8 - 有区别，计算nP.unique(ip_cnt)
        final_access['count_diff_request'] = 0 #9
        final_access['count_loss'] = 0 #11
        final_access['count_hour_loss'] = 0 #新增

        #新增：
        # final_access['ip_req_domain']=0

        add_attr_len = len(add_list)
        str_list_len = len(str_list)

        total_dif_ip = train_access[['fqdn_no','encoded_ip']]

        for i in tqdm(range(len(final_access))):
            temp_df = train_access[train_access['fqdn_no']==final_access.iloc[i,0]]
            count_diff_ip  = list(temp_df.drop_duplicates(['encoded_ip'])['encoded_ip'].to_list())
            #按小时聚合
            tmp_hour_df = temp_df.loc[:,['count','hour','date']].groupby(['hour','date'])['count'].sum().reset_index(name='count_hour_req')
            count_diff_request = len(tmp_hour_df.drop_duplicates(['count_hour_req'])['count_hour_req'].to_list())
            count_day_loss = day_len -  len(temp_df.drop_duplicates(['date'])['date'].to_list())
            count_hour_loss = 24 -  len(temp_df.drop_duplicates(['hour']))
        #     pure_tmp_df =  temp_df[['fqdn_no','encoded_ip']]
        #     pure_no_df =pure_df[pure_df['fqdn_no']!=final_access.iloc[i,0]]
        #     ip_req_domain = len(pd.merge(pure_tmp_df, pure_no_df ,on='encoded_ip'))
            ip_rec_domain = temp_df.groupby(['fqdn_no', 'encoded_ip'])['count'].sum().reset_index(name='ip_rec_domain')[
                'ip_rec_domain'].to_list()
            ip_tmp_list = list(temp_df.drop_duplicates(['encoded_ip'])['encoded_ip'])
            tmp_df2 = train_access[train_access['encoded_ip'].isin(ip_tmp_list)]
            ip_related_list = tmp_df2.groupby('encoded_ip')['fqdn_no'].count().reset_index(name='ip_related_list')['ip_related_list'].to_list()
            for j in range(str_list_len):
                if j < str_list_len - 2:
                    v1 = final_access.iloc[i, 1 + day_len * j:1 + day_len * (j + 1)]
                    if (v1.max() > 0):


                        # org_s = v1.shape

                        if(j!=1):
                            v1 = np.array(list(v1))
                            v1 = np.array(list(v1))
                            v1 = v1[~np.isnan(v1)]
                            v1 = v1[v1 != 0]
                            v1 = pd.Series(v1)
                        else:
                            v1.fillna(0, inplace=True)
                            v1.loc[len(v1)]=0
                        # del_s = v1.shape
                        # if(org_s!=del_s):
                        #     print(org_s[0]-del_s[0])

                        final_access.iloc[i, len_df1 + j * add_attr_len] = v1.sum()


                        final_access.iloc[i, len_df1 + j * add_attr_len + 1] = v1.var(ddof=0)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 2] = v1.std(ddof=0)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 3] = v1.mean()
                        final_access.iloc[i, len_df1 + j * add_attr_len + 4] = v1.max()
                        final_access.iloc[i, len_df1 + j * add_attr_len + 5] = v1.min()
                        # tmp_3 = np.percentile(v1, [25, 50, 75])
                        final_access.iloc[i, len_df1 + j * add_attr_len + 6] = v1.quantile(0.25)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 7] = v1.quantile(0.50)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 8] = v1.quantile(0.75)

                else:
                    if j == str_list_len - 2:
                        v1 = ip_rec_domain
                    if j == str_list_len - 1:
                        v1 = ip_related_list
                    v1 = np.array(v1)
                    v1 = v1[~np.isnan(v1)]
                    v1 = list(v1)
                    if (len(v1) > 1):
                        # list1 = ['sum','var','std','mean','max','min','percent25','median','percent75']
                        final_access.iloc[i, len_df1 + j * add_attr_len] = np.sum(v1)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 1] = np.var(v1)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 2] = np.std(v1)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 3] = np.mean(v1)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 4] = np.max(v1)
                        final_access.iloc[i, len_df1 + j * add_attr_len + 5] = np.min(v1)
                        tmp_3 = np.percentile(v1, [25, 50, 75])
                        final_access.iloc[i, len_df1 + j * add_attr_len + 6] = int(tmp_3[0])
                        final_access.iloc[i, len_df1 + j * add_attr_len + 7] = int(tmp_3[1])
                        final_access.iloc[i, len_df1 + j * add_attr_len + 8] = int(tmp_3[2])



            final_access.loc[i,'count_ip'] = np.sum(temp_df['count'].to_list())
            final_access.loc[i,'count_diff_ip'] = len(count_diff_ip)
            final_access.loc[i,'count_diff_request'] = count_diff_request
            final_access.loc[i,'count_loss'] = count_day_loss
            final_access.loc[i,'count_hour_loss'] = count_hour_loss



        access_attr = final_access.columns.values.tolist()
        access_feature = final_access.drop(access_attr[1:len_df1],axis=1)
        access_feature.to_csv(r'Feature_Extract/02_Feature_SRC/access_feature.csv',index=False)




# In[ ]:




