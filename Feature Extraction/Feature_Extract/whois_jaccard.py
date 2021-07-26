#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import json
from tqdm import tqdm


class WHOISJaccard:
    def __init__(self,whois_path=r'Data/whois.json',path_train_fqdn=r'Data/fqdn.csv',path_df_label=r'Feature_Extract/Label_For_Feature_Extract/label.csv'):


        json_fqdn_list=np.load('Feature_Extract/json_fqdn_list.npy',allow_pickle=True)
        json_fqdn_list=json_fqdn_list.tolist()


        # In[3]:


        label_mali = list(range(0,9)) #[1]
        label_str = 'family_no'#'label'

        df_label = pd.read_csv(path_df_label)


        # In[4]:


        label_fqdn_list = []
        for mali_no in range(9):
            tmp_fqdn = df_label[df_label[label_str].isin([mali_no])]['fqdn_no'].to_list()
            label_fqdn_list.append(tmp_fqdn)


        # In[13]:


        #取fqdn_no的no构成list
        label_no_list = []
        for i in range(len(label_fqdn_list)):
            tmp_list = []
            tmp = label_fqdn_list[i]
            for j in range(len( tmp)):
                tmp_list.append(int(tmp[j].split('_')[-1]))
            label_no_list.append(tmp_list)


        # In[19]:


        dict_list  = list(json_fqdn_list[0][1])
        dict_list


        # In[14]:



        train_fqdn = pd.read_csv(path_train_fqdn)
        fqdn_list = list(train_fqdn['fqdn_no'])


        # In[39]:


        mali_info_list = []
        for i in range(9):
            ns_list = []
            admin_email = []
            registrant_email = []
            tech_email =[]
            r_whoisserver_list=[]
            whoisserver = []
            sponsoring = []
            tmp_list = []
            for j in tqdm(range(len(label_no_list[i]))):
                fqdn = label_no_list[i][j]
                tmp_whois_info = json_fqdn_list[fqdn]
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
                ns_list.extend(tmp_ns_list)
                admin_email.extend(tmp_admin_email)
                registrant_email.extend(tmp_registrant_email)
                tech_email.extend(tmp_tech_email)
                r_whoisserver_list.extend(tmp_r_whoisserver_list)
                whoisserver.extend(tmp_whoisserver )
                sponsoring .extend(tmp_sponsoring )
            tmp_list.append(list(set(filter(None, ns_list))))
            tmp_list.append(list(set(filter(None, admin_email))))
            tmp_list.append(list(set(filter(None, registrant_email))))
            tmp_list.append(list(set(filter(None, tech_email))))
            tmp_list.append(list(set(filter(None, r_whoisserver_list))))
            tmp_list.append(list(set(filter(None, whoisserver))))
            tmp_list.append(list(set(filter(None, sponsoring))))
            mali_info_list.append(tmp_list)


        # In[40]:


        mali_info_list


        # In[61]:


        len(mali_info_list[8][1])


        # In[60]:


        len(set(mali_info_list[8][1])-set(mali_info_list[0][1]))


        # In[64]:








        # In[72]:


        df=pd.DataFrame()
        df['fqdn_no'] = fqdn_list


        # In[73]:


        df


        # In[69]:


        def jaccard_count(list1,list2):
            m = len(set(list1).union(set(list2)))
            n = len(set(list1).intersection(set(list2)))
            return (n/m)


        # In[74]:


        estimate_list = [ 'ns','admin_email','registrant','tech_email','r_whoisserver','whoisserver','sponsoring']
        for i in range(9):
            for j in range(len(estimate_list)):
                df[str(estimate_list[j]+'_'+str(i))] = 0



        for i in tqdm(range(len(df))):
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
            df.iloc[i,1] = jaccard_count(mali_info_list[0][0],tmp_ns_list)
            df.iloc[i,8] = jaccard_count(mali_info_list[1][0],tmp_ns_list)
            df.iloc[i,15] = jaccard_count(mali_info_list[2][0],tmp_ns_list)
            df.iloc[i,22] = jaccard_count(mali_info_list[3][0],tmp_ns_list)
            df.iloc[i,29] = jaccard_count(mali_info_list[4][0],tmp_ns_list)
            df.iloc[i,36] = jaccard_count(mali_info_list[5][0],tmp_ns_list)
            df.iloc[i,43] = jaccard_count(mali_info_list[6][0],tmp_ns_list)
            df.iloc[i,50] = jaccard_count(mali_info_list[7][0],tmp_ns_list)
            df.iloc[i,57] = jaccard_count(mali_info_list[8][0],tmp_ns_list)
            tmp_admin_email = (list(set(filter(None, tmp_admin_email))))
            df.iloc[i,2] = jaccard_count(mali_info_list[0][1],tmp_admin_email)
            df.iloc[i,9] = jaccard_count(mali_info_list[1][1],tmp_admin_email)
            df.iloc[i,16] = jaccard_count(mali_info_list[2][1],tmp_admin_email)
            df.iloc[i,23] = jaccard_count(mali_info_list[3][1],tmp_admin_email)
            df.iloc[i,30] = jaccard_count(mali_info_list[4][1],tmp_admin_email)
            df.iloc[i,37] = jaccard_count(mali_info_list[5][1],tmp_admin_email)
            df.iloc[i,44] = jaccard_count(mali_info_list[6][1],tmp_admin_email)
            df.iloc[i,51] = jaccard_count(mali_info_list[7][1],tmp_admin_email)
            df.iloc[i,58] = jaccard_count(mali_info_list[8][1],tmp_admin_email)
            tmp_registrant_email = (list(set(filter(None, tmp_registrant_email))))
            df.iloc[i,3] = jaccard_count(mali_info_list[0][2],tmp_registrant_email)
            df.iloc[i,10] = jaccard_count(mali_info_list[1][2],tmp_registrant_email)
            df.iloc[i,17] = jaccard_count(mali_info_list[2][2],tmp_registrant_email)
            df.iloc[i,24] = jaccard_count(mali_info_list[3][2],tmp_registrant_email)
            df.iloc[i,31] = jaccard_count(mali_info_list[4][2],tmp_registrant_email)
            df.iloc[i,38] = jaccard_count(mali_info_list[5][2],tmp_registrant_email)
            df.iloc[i,45] = jaccard_count(mali_info_list[6][2],tmp_registrant_email)
            df.iloc[i,52] = jaccard_count(mali_info_list[7][2],tmp_registrant_email)
            df.iloc[i,59] = jaccard_count(mali_info_list[8][2],tmp_registrant_email)
            tmp_tech_email = (list(set(filter(None, tmp_tech_email))))
            df.iloc[i,4] = jaccard_count(mali_info_list[0][3],tmp_tech_email)
            df.iloc[i,11] = jaccard_count(mali_info_list[1][3],tmp_tech_email)
            df.iloc[i,18] = jaccard_count(mali_info_list[2][3],tmp_tech_email)
            df.iloc[i,25] = jaccard_count(mali_info_list[3][3],tmp_tech_email)
            df.iloc[i,32] = jaccard_count(mali_info_list[4][3],tmp_tech_email)
            df.iloc[i,39] = jaccard_count(mali_info_list[5][3],tmp_tech_email)
            df.iloc[i,46] = jaccard_count(mali_info_list[6][3],tmp_tech_email)
            df.iloc[i,53] = jaccard_count(mali_info_list[7][3],tmp_tech_email)
            df.iloc[i,60] = jaccard_count(mali_info_list[8][3],tmp_tech_email)
            tmp_r_whoisserver_list = (list(set(filter(None, tmp_r_whoisserver_list))))
            df.iloc[i,5] = jaccard_count(mali_info_list[0][4],tmp_r_whoisserver_list)
            df.iloc[i,12] = jaccard_count(mali_info_list[1][4],tmp_r_whoisserver_list)
            df.iloc[i,19] = jaccard_count(mali_info_list[2][4],tmp_r_whoisserver_list)
            df.iloc[i,26] = jaccard_count(mali_info_list[3][4],tmp_r_whoisserver_list)
            df.iloc[i,33] = jaccard_count(mali_info_list[4][4],tmp_r_whoisserver_list)
            df.iloc[i,40] = jaccard_count(mali_info_list[5][4],tmp_r_whoisserver_list)
            df.iloc[i,47] = jaccard_count(mali_info_list[6][4],tmp_r_whoisserver_list)
            df.iloc[i,54] = jaccard_count(mali_info_list[7][4],tmp_r_whoisserver_list)
            df.iloc[i,61] = jaccard_count(mali_info_list[8][4],tmp_r_whoisserver_list)
            tmp_whoisserver = (list(set(filter(None, tmp_whoisserver))))
            df.iloc[i,6] = jaccard_count(mali_info_list[0][5],tmp_whoisserver)
            df.iloc[i,13] = jaccard_count(mali_info_list[1][5],tmp_whoisserver)
            df.iloc[i,20] = jaccard_count(mali_info_list[2][5],tmp_whoisserver)
            df.iloc[i,27] = jaccard_count(mali_info_list[3][5],tmp_whoisserver)
            df.iloc[i,34] = jaccard_count(mali_info_list[4][5],tmp_whoisserver)
            df.iloc[i,41] = jaccard_count(mali_info_list[5][5],tmp_whoisserver)
            df.iloc[i,48] = jaccard_count(mali_info_list[6][5],tmp_whoisserver)
            df.iloc[i,55] = jaccard_count(mali_info_list[7][5],tmp_whoisserver)
            df.iloc[i,62] = jaccard_count(mali_info_list[8][5],tmp_whoisserver)
            tmp_sponsoring = (list(set(filter(None,tmp_sponsoring))))
            df.iloc[i,7] = jaccard_count(mali_info_list[0][6],tmp_sponsoring)
            df.iloc[i,14] = jaccard_count(mali_info_list[1][6],tmp_sponsoring)
            df.iloc[i,21] = jaccard_count(mali_info_list[2][6],tmp_sponsoring)
            df.iloc[i,28] = jaccard_count(mali_info_list[3][6],tmp_sponsoring)
            df.iloc[i,35] = jaccard_count(mali_info_list[4][6],tmp_sponsoring)
            df.iloc[i,42] = jaccard_count(mali_info_list[5][6],tmp_sponsoring)
            df.iloc[i,49] = jaccard_count(mali_info_list[6][6],tmp_sponsoring)
            df.iloc[i,56] = jaccard_count(mali_info_list[7][6],tmp_sponsoring)
            df.iloc[i,63] = jaccard_count(mali_info_list[8][6],tmp_sponsoring)





        df.to_csv(r'Feature_Extract/02_Feature_SRC/whois_jaccard.csv',index=False)





