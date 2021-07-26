import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from feature_selection import PIFS
from sklearn.metrics import *

def generate_pudata(seed,data_path,label_frequency):
    rdg = np.random.RandomState(seed)
    df_all = pd.read_csv(data_path)
    columns_list = df_all.columns.to_list()
    feature_list = columns_list[1:len(columns_list)-1]

    df_mali = df_all[df_all['family_no'].isin(range(9))]
    mali_no = list(df_mali.family_no.value_counts().reset_index(name='counts')['index'])
    mali_count = list(df_mali.family_no.value_counts().reset_index(name='counts')['counts'])
    df_P = pd.DataFrame(columns=df_all.columns.to_list())
    #随机取label_frequency的P
    for no, num in zip(mali_no, mali_count):
        cut_num =int(np.ceil(num * label_frequency))
        #print("family_no", no, ":", cut_num)
        df_indices = pd.DataFrame(rdg.permutation(df_mali[df_mali['family_no']==no])[:cut_num], columns=columns_list )
        df_P = pd.concat([df_P,df_indices],axis=0)
    P_list = df_P.fqdn_no.to_list()
    df_U = df_all[~df_all['fqdn_no'].isin(P_list)]

    #dataFrame转numpy格式，并随机打乱顺序
    df_P = pd.DataFrame(rdg.permutation(df_P),columns=columns_list )
    data_P = np.array(df_P.iloc[:,1:len(feature_list)+1])
    label_P = np.array(df_P.family_no.to_list())

    df_U = pd.DataFrame(rdg.permutation(df_U),columns=columns_list )
    data_U = np.array(df_U.iloc[:,1:len(feature_list)+1])
    label_U = np.array(df_U.family_no.to_list())
    pred_U = []
    for c in  label_U :
        if c == 9:
            pred_U.append(0)
        else :
            pred_U.append(1)
    pred_U = np.array(pred_U)


    return data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list

rf = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018)


def feature_selection(seed,data_path,label_frequency):
    start = time.time()
    data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list = generate_pudata(seed,data_path,label_frequency)
    dataTrain = np.r_[data_P,data_U]
    labelTrain = np.r_[[1]*len(data_P),[0]*len(data_U)]
    dataValid = data_U
    labelValid = pred_U
    fs = PIFS(rf,dataTrain,labelTrain,dataValid,labelValid,feature_list,flag ='pumd')


    new_baseline, new_feature_list, feature_index = fs.mainFS()
    print(len(feature_index))
    print(new_baseline)
    df_all = pd.read_csv(data_path)
    new_feature_list.insert(0,'fqdn_no')
    new_feature_list.append('family_no')
    data_all = df_all[new_feature_list]
    data_all.to_csv(r'new_feature_list.csv',index=False)






    decription = 'The selection of feature list has been completed.'
    print(decription, 'Running_Time:{:.2f}s\n'.format(time.time()-start))


    return 0




if __name__ == '__main__':
    file_name = #original_feature_list include 'fqdn_no' and 'family_no'
    data_path = r'SRC_feature/'+file_name
    df_value = feature_selection(100, data_path, 0.5)



