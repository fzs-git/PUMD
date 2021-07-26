#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm


# **②  基于flint.csv 中的特征：**   
#   
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
#  
# **③  基于access.csv 中的特征：**   
#   
# - 14. <font color=red>**jaccard_ip_no**</font>：计算域的请求集IP与各恶意域家族的Jaccard相似系数 [244,261]
#     - 举例：
#         查询域A的IP列表【IP1，IP2，IP3】
#         查询域B的IP列表【IP2，IP3，IP4】
#         Jaccard系数：相同交集IP2，IP3=2个，并集IP1/2/3/4=4个，则得到相似系数 2/4 =0.5
#         实际使用统计各个恶意家族的请求集IP，计算待预测域与各恶意家族IP列表的jaccard系数  
#   
# **④  基于ip/ipv6.csv 中的特征：** 
# 
# 3. subvision_jaccard_no: 统计一个域映射的IP所属地区与恶意域的IP所属地区相似性[147,155]
# - 举例：
#     域A映射的IP所属地区【‘北京’，‘上海’，‘天津’】
#     域A映射的IP所属地区【‘北京’，‘上海’，‘湖南’】
#     Jaccard系数：相同交集北京’，‘上海’=2个，并集‘北京’，‘上海’，‘天津’，‘湖南’=4个，则得到相似系数 2/4 =0.5
#     实际使用统计与带标签的良性/恶意域的IP所属区域集，计算待预测域与带标签的良性/恶意域的IP所属区域列表的jaccard系数
# 
# **⑤  基于whois.json 中的特征：** 
# 

# In[2]:
class JaccardFeature:
    def __init__(self,path_df_label=r'Feature_Extract/Label_For_Feature_Extract/label.csv',
                 path_train_fqdn=r'Data/fqdn.csv',path_train_flint = r'Data/flint.csv',
                 path_train_access=r'Data/access.csv',path_train_ip=r'Data/ip.csv',path_train_ipv6=r'Data/ipv6.csv'):

        df_label = pd.read_csv(path_df_label)
        train_flint = pd.read_csv(path_train_flint)
        train_fqdn = pd.read_csv(path_train_fqdn)
        train_access = pd.read_csv(path_train_access)
        train_ip = pd.read_csv(path_train_ip)
        train_ipv6 = pd.read_csv(path_train_ipv6)
        label_str = 'family_no'
        label_mali = list(np.unique(list(df_label[label_str])))

        def jaccard_count(list1,list2):
            m = len(set(list1).union(set(list2)))
            n = len(set(list1).intersection(set(list2)))
            if n>0:
                return (n/m)
            else:
                return 0

        def make_attrib_no(str1,str2,df,family_no):

            for i in family_no:
                df[str1+str(i)+str2]=0

        train_ip = train_ip.loc[:,['encoded_ip','country','subdivision']]
        train_ipv6 = train_ipv6.loc[:,['encoded_ip','country','subdivision']]
        train_ip = pd.concat([train_ipv6,train_ip],axis=0)
        tmp_flint = train_flint.drop_duplicates(['fqdn_no_x','encoded_value'])[['fqdn_no_x','encoded_value']]
        train_df_ip = pd.merge(tmp_flint,train_ip,left_on='encoded_value',right_on = 'encoded_ip')

        train_df_ip.rename(columns = {'fqdn_no_x':'fqdn_no'},inplace=True)

        label_feature = pd.DataFrame()
        label_feature['fqdn_no'] = train_fqdn['fqdn_no'].to_list()
        mali_no_fqdn = []
        mali_fqdn = df_label[df_label[label_str].isin(label_mali)]['fqdn_no'].to_list()
        mali_access = train_access[train_access['fqdn_no'].isin(mali_fqdn)]
        #flint
        mali_no_rr = []
        mali_no_ip = []
        mali_no_cname = []
        mali_no_ns = []
        mali_no_mx = []
        # # df赋值
        make_attrib_no('jaccard_rr_','',label_feature,label_mali)
        make_attrib_no('jaccard_ip_','',label_feature,label_mali)
        make_attrib_no('jaccard_cname_','',label_feature,label_mali)
        make_attrib_no('jaccard_ns_','',label_feature,label_mali)
        make_attrib_no('jaccard_mx_','',label_feature,label_mali)
        make_attrib_no('ip_','_counts',label_feature,label_mali)
        make_attrib_no('ip_','_dif_counts',label_feature,label_mali)
        label_feature['ip_mali_counts']=0
        label_feature['ip_mali_dif_counts']=0
        label_feature['jaccard_rr_mali']=0
        label_feature['jaccard_cname_mali']=0
        label_feature['jaccard_ns_mali']=0
        label_feature['jaccard_mx_mali']=0
        label_feature['jaccard_ip_mali']=0
        #access
        jaccard_ip_access_no = []
        # # df赋值
        make_attrib_no('jaccard_ip_access_','',label_feature,label_mali)
        #ip/ipv6
        subvision_jaccard_no = []#no_cnt
        mali_subvision_list = []#全部
        # # df赋值
        label_feature['subvision_jaccard_mali'] = 0
        make_attrib_no('subvision_jaccard_','',label_feature,label_mali)
        for mali_no in label_mali:

            tmp_fqdn = df_label[df_label[label_str].isin([mali_no])]['fqdn_no'].to_list()
            mali_no_fqdn.append(tmp_fqdn)
            #access
            tmp_ip_list =train_access[train_access['fqdn_no'].isin(tmp_fqdn)].drop_duplicates(['encoded_ip'])['encoded_ip'].to_list()

            jaccard_ip_access_no.append(tmp_ip_list)
            #flint
            tmp_flint = train_flint[train_flint['fqdn_no_x'].isin(tmp_fqdn)]
            mali_flint = tmp_flint.loc[:,['fqdn_no_x','flintType','encoded_value']].drop_duplicates()
            # rr
            mali_value = np.unique(mali_flint['encoded_value'].to_list())
            mali_no_rr.append(mali_value)
            #ip
            tmp_mali_flint_ip = mali_flint[mali_flint['flintType'].isin([1,28])]
            mali_ip_value = np.unique(tmp_mali_flint_ip['encoded_value'].to_list())
            mali_no_ip .append(mali_ip_value)
            #cname
            mali_cname_value = np.unique(mali_flint[mali_flint['flintType'].isin([5])]['encoded_value'].to_list())
            mali_no_cname.append( mali_cname_value )
            #ns
            mali_ns_value = np.unique(mali_flint[mali_flint['flintType'].isin([2])]['encoded_value'].to_list())
            mali_no_ns.append(mali_ns_value)
            #mx
            mali_mx_value = np.unique(mali_flint[mali_flint['flintType'].isin([15])]['encoded_value'].to_list())
            mali_no_mx.append(mali_mx_value)
            #ip/ipv6
            subvision_jaccard_no.append(np.unique(train_df_ip[train_df_ip['fqdn_no'].isin(tmp_fqdn)]['subdivision'].to_list()))
            mali_subvision_list.extend(np.unique(train_df_ip[train_df_ip['fqdn_no'].isin(tmp_fqdn)]['subdivision'].to_list()))
        #flint
        # for i in range(1,9):
        #     print(len(jaccard_ip_access_no[i]))
        mali_fqdn = df_label[df_label[label_str].isin(label_mali)]['fqdn_no'].to_list()
        mali_flint = train_flint[train_flint['fqdn_no_x'].isin(mali_fqdn)]
        mali_flint = mali_flint.loc[:,['fqdn_no_x','flintType','encoded_value']].drop_duplicates()
        mali_flint_ip = mali_flint[mali_flint['flintType'].isin([1,28])]
        mali_value = np.unique(mali_flint['encoded_value'].to_list())
        mali_ip_value = np.unique(mali_flint_ip['encoded_value'].to_list())
        mali_cname_value = np.unique(mali_flint[mali_flint['flintType'].isin([5])]['encoded_value'].to_list())
        mali_ns_value = np.unique(mali_flint[mali_flint['flintType'].isin([2])]['encoded_value'].to_list())
        mali_mx_value = np.unique(mali_flint[mali_flint['flintType'].isin([15])]['encoded_value'].to_list())

        for i in tqdm(range(len(label_feature))):

            #flint
            tmp_df= train_flint[train_flint['fqdn_no_x']==label_feature.iloc[i,0]]
            #access
            temp_access_df = train_access[train_access['fqdn_no']==label_feature.iloc[i,0]]
            #ip/ipv6
            tmp_ip_df = train_df_ip[train_df_ip['fqdn_no'] == label_feature.iloc[i,0]]
            if len(temp_access_df) > 0:

                tmp_ip_access_list = temp_access_df.drop_duplicates(['encoded_ip'])['encoded_ip'].to_list()

            for mali_no in label_mali:
                # flint
                if len(tmp_df)>0:
                    tmp_df_drop = tmp_df.loc[:,['fqdn_no_x','flintType','encoded_value']].drop_duplicates()
                    merge_tmp = pd.merge(tmp_df_drop,mali_flint_ip,on='encoded_value')
                    label_feature.loc[i,'ip_mali_counts'] = len(merge_tmp)
                    label_feature.loc[i,'ip_mali_dif_counts'] = len(np.unique(merge_tmp['fqdn_no_x_y'].to_list()))
                    label_feature.loc[i,'jaccard_rr_mali'] = jaccard_count(tmp_df['encoded_value'].to_list(),mali_value)
                    label_feature.loc[i,'jaccard_cname_mali']=jaccard_count(tmp_df['encoded_value'].to_list(),mali_cname_value)
                    label_feature.loc[i,'jaccard_ns_mali']=jaccard_count(tmp_df['encoded_value'].to_list(),mali_ns_value)
                    label_feature.loc[i,'jaccard_mx_mali']=jaccard_count(tmp_df['encoded_value'].to_list(),mali_mx_value)
                    label_feature.loc[i,'jaccard_ip_mali']=jaccard_count(tmp_df['encoded_value'].to_list(),mali_ip_value)
                    label_feature.loc[i,'subvision_jaccard_mali']=jaccard_count(tmp_ip_df['subdivision'].to_list(),mali_subvision_list)



                    tmp_mali_flint_ip = mali_flint_ip[mali_flint_ip['fqdn_no_x'].isin(mali_no_fqdn[mali_no])]
                    tmp_merge_tmp = pd.merge(tmp_df_drop,tmp_mali_flint_ip,on='encoded_value')
                    label_feature.loc[i,'ip_'+str(mali_no)+'_counts'] = len(tmp_merge_tmp)
                    label_feature.loc[i,'ip_'+str(mali_no)+'_dif_counts'] = len(np.unique(tmp_merge_tmp['fqdn_no_x_y'].to_list()))
                    label_feature.loc[i,'jaccard_rr_'+str(mali_no)] = jaccard_count(tmp_df['encoded_value'].to_list(),mali_no_rr[mali_no])
                    label_feature.loc[i,'jaccard_ip_'+str(mali_no)]=jaccard_count(tmp_df['encoded_value'].to_list(),mali_no_ip[mali_no])
                    label_feature.loc[i,'jaccard_cname_'+str(mali_no)]=jaccard_count(tmp_df['encoded_value'].to_list(),mali_no_cname[mali_no])
                    label_feature.loc[i,'jaccard_ns_'+str(mali_no)]=jaccard_count(tmp_df['encoded_value'].to_list(),mali_no_ns[mali_no])
                    label_feature.loc[i,'jaccard_mx_'+str(mali_no)]=jaccard_count(tmp_df['encoded_value'].to_list(),mali_no_mx[mali_no])
                #access
                if len(temp_access_df)>0:

                    label_feature.loc[i,'jaccard_ip_access_'+str(mali_no)] = jaccard_count(tmp_ip_access_list,jaccard_ip_access_no[mali_no])
                    # print(jaccard_count(tmp_ip_access_list,jaccard_ip_access_no[mali_no]))
                #ip/ipv6
                if len(tmp_ip_df) > 0:
                    label_feature.loc[i,'subvision_jaccard_'+str(mali_no)]= jaccard_count(np.unique(tmp_ip_df['subdivision'].to_list()),subvision_jaccard_no[mali_no])

        # label_feature.to_csv(r'Feature_Extract/tmp_jaccfeature.csv',index=False)
        # return

        json_fqdn_list=np.load('Feature_Extract/json_fqdn_list.npy',allow_pickle=True)
        json_fqdn_list=json_fqdn_list.tolist()
        dict_list  = list(json_fqdn_list[0][1])
        #取fqdn_no的no构成list
        label_no_list = []
        for i in range(len(mali_no_fqdn)):
            tmp_list = []
            tmp = mali_no_fqdn[i]
            for j in range(len(tmp)):
                tmp_list.append(int(tmp[j].split('_')[-1]))
            label_no_list.append(tmp_list)

        mali_info_list = []
        for i in range(9):
            ns_list = []
            admin_email = []
            registrant_email = []
            tech_email = []
            r_whoisserver_list = []
            whoisserver = []
            sponsoring = []
            tmp_list = []
            for j in tqdm(range(len(label_no_list[i]))):
                fqdn = label_no_list[i][j]
                tmp_whois_info = json_fqdn_list[fqdn]
                # tmp_list
                tmp_ns_list = []
                tmp_admin_email = []
                tmp_registrant_email = []
                tmp_tech_email = []
                tmp_r_whoisserver_list = []
                tmp_whoisserver = []
                tmp_sponsoring = []
                for m in range(len(tmp_whois_info)):
                    if (tmp_whois_info[m]['nameservers'] is not None):
                        tmp_ns_list.extend(tmp_whois_info[m]['nameservers'])
                    if (tmp_whois_info[m]['admin_email'] is not None):
                        tmp_admin_email.extend(tmp_whois_info[m]['admin_email'])
                    if (tmp_whois_info[m]['registrant_email'] is not None):
                        tmp_registrant_email.extend(tmp_whois_info[m]['registrant_email'])
                    if (tmp_whois_info[m]['tech_email'] is not None):
                        tmp_tech_email.extend(tmp_whois_info[m]['tech_email'])
                    if (tmp_whois_info[m]['r_whoisserver_list'] is not None):
                        tmp_r_whoisserver_list.extend(tmp_whois_info[m]['r_whoisserver_list'])

                    tmp_whoisserver.append(tmp_whois_info[m]['whoisserver'])
                    tmp_sponsoring.append(tmp_whois_info[m]['sponsoring'])
                ns_list.extend(tmp_ns_list)
                admin_email.extend(tmp_admin_email)
                registrant_email.extend(tmp_registrant_email)
                tech_email.extend(tmp_tech_email)
                r_whoisserver_list.extend(tmp_r_whoisserver_list)
                whoisserver.extend(tmp_whoisserver)
                sponsoring.extend(tmp_sponsoring)
            tmp_list.append(list(set(filter(None, ns_list))))
            tmp_list.append(list(set(filter(None, admin_email))))
            tmp_list.append(list(set(filter(None, registrant_email))))
            tmp_list.append(list(set(filter(None, tech_email))))
            tmp_list.append(list(set(filter(None, r_whoisserver_list))))
            tmp_list.append(list(set(filter(None, whoisserver))))
            tmp_list.append(list(set(filter(None, sponsoring))))
            mali_info_list.append(tmp_list)

        estimate_list = [ 'ns','admin_email','registrant','tech_email','r_whoisserver','whoisserver','sponsoring']
        for l_name in estimate_list:
            make_attrib_no(l_name +'_','',label_feature,label_mali)

        label_feature = label_feature.iloc[:20512,:]

        for i in tqdm(range(len(label_feature))):
            tmp_whois_info = json_fqdn_list[i]
            #tmp_list
            tmp_ns_list = []
            tmp_admin_email = []
            tmp_registrant_email = []
            tmp_tech_email =[]
            tmp_r_whoisserver_list=[]
            tmp_whoisserver = []
            tmp_sponsoring = []
            for m in range(len(tmp_whois_info)):
                if(tmp_whois_info[m]['nameservers']is not None):
                    tmp_ns_list.extend(tmp_whois_info[m]['nameservers'])
                if(tmp_whois_info[m]['admin_email']is not None):
                    tmp_admin_email.extend(tmp_whois_info[m]['admin_email'])
                if(tmp_whois_info[m]['registrant_email']is not None):
                    tmp_registrant_email.extend(tmp_whois_info[m]['registrant_email'])
                if(tmp_whois_info[m]['tech_email']is not None):
                    tmp_tech_email.extend(tmp_whois_info[m]['tech_email'])
                if(tmp_whois_info[m]['r_whoisserver_list']is not None):
                    tmp_r_whoisserver_list.extend(tmp_whois_info[m]['r_whoisserver_list'])

                tmp_whoisserver .append(tmp_whois_info[m]['whoisserver'])
                tmp_sponsoring.append(tmp_whois_info[m]['sponsoring'])

            tmp_ns_list = (list(set(filter(None,  tmp_ns_list))))
            tmp_admin_email = (list(set(filter(None, tmp_admin_email))))
            tmp_registrant_email = (list(set(filter(None, tmp_registrant_email))))
            tmp_tech_email = (list(set(filter(None, tmp_tech_email))))
            tmp_r_whoisserver_list = (list(set(filter(None, tmp_r_whoisserver_list))))
            tmp_whoisserver = (list(set(filter(None, tmp_whoisserver))))
            tmp_sponsoring = (list(set(filter(None,tmp_sponsoring))))
            for mali_no in label_mali:
                label_feature.loc[i,'ns_'+str(mali_no)] = jaccard_count(mali_info_list[mali_no][0],tmp_ns_list)
                label_feature.loc[i,'admin_email_'+str(mali_no)] = jaccard_count(mali_info_list[mali_no][1],tmp_admin_email)
                label_feature.loc[i,'registrant_'+str(mali_no)] = jaccard_count(mali_info_list[mali_no][2],tmp_registrant_email)
                label_feature.loc[i,'tech_email_'+str(mali_no)] = jaccard_count(mali_info_list[mali_no][3],tmp_tech_email)
                label_feature.loc[i,'r_whoisserver_'+str(mali_no)] = jaccard_count(mali_info_list[mali_no][4],tmp_r_whoisserver_list)
                label_feature.loc[i,'whoisserver_'+str(mali_no)] = jaccard_count(mali_info_list[mali_no][5],tmp_whoisserver)
                label_feature.loc[i,'sponsoring_'+str(mali_no)] = jaccard_count(mali_info_list[mali_no][6],tmp_sponsoring)

        label_feature.to_csv(r'Feature_Extract/02_Feature_SRC/jaccard_Feature.csv',index=False)
