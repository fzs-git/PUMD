import pandas as pd
import numpy as np
import os
class FEATUREUNION:
    def __init__(self):
        path = r'Feature_Extract/02_Feature_SRC'
        paths = os.listdir(path)
        df_com = pd.DataFrame()
        len_all = 0
        print("Feature Union:")
        for i in paths:
            com_path = os.path.join(path,i)
            tmd_df = pd.read_csv(com_path)
            tmp_len = len(tmd_df.columns.to_list())
            print("Append Feature:",tmp_len)
        #     print(tmp_len)
            if(len(df_com)>1):
                df_com = pd.merge(df_com,tmd_df,on='fqdn_no')
                len_all =  len_all + tmp_len - 1
            else:
                df_com = tmd_df
                len_all = tmp_len
        print("Total number of feature sets:",len_all)
        df_com.to_csv(r'Feature_Extract/feature_all.csv',index=False)