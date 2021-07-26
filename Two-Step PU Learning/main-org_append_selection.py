import numpy as np
import time
import pandas as pd
import copy

from xgboost import XGBClassifier
from PUMD import PUMD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
# matthews_corrcoef
from sklearn.svm import SVC
import os
import matplotlib.pylab as plt
import time


def model_performance(y_pred,y_true,y_prob=False):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn, fp, fn, tp)
    recall = tp / (tp + fn)  #tp/p
    specificity = tn / (tn + fp)   #tn/n
    precision = tp/(tp+fp)
    gmean = np.sqrt(recall * specificity)
    f_score = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)


    if np.array(y_prob).any():
        auc = roc_auc_score(y_true, y_prob)
        performance = [f_score, gmean, recall,precision, acc,auc,mcc]
        # fpr, tpr, thersholds = roc_curve(y_true, y_pred)

        # precision, recall, th = precision_recall_curve(y_true, y_prob)
        # f1s = 2 * precision[:-1] * recall[:-1]/(precision[:-1] + recall[:-1])
        # best_th = np.argmax(f1s)
        # print('best_fq:',np.max(f1s))

        # plt.figure(1)
        # plt.plot(fpr, tpr)
        # plt.xlabel('fpr')
        # plt.ylabel('tpr')
        # plt.show()


    else:
        performance = [f_score, gmean, recall, precision, acc,mcc]
    return np.array(performance)


def generate_feature_select(df,delete_list):
    # print(len(delete_list))
    df = df.drop(delete_list, axis=1)
    # print(len(df.columns))
    return df
def feature_select_func(np_org,delete_list,axis=1):
    np_fs = np.delete(np_org,delete_list,axis)
    return np_fs

def generate_pudata(seed,data_path,label_frequency):
    rdg = np.random.RandomState(seed)
    df_all = pd.read_csv(data_path)
    # delete_feature = list(np.load(r'D:\Project\Python\data\new-4-6\Intersection_list_ex.npy'))
    # delete_feature = list(np.load(r'D:\Project\Python\data\new-4-6\Intersection_list.npy'))
    delete_feature = []
    df_all = generate_feature_select(df_all,delete_feature)
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
    # print('P:',len(data_P))
    # print('U:',len(data_U))
    # print('benign:',len(df_all)-len(df_mali))
    # print('mali:',  len(df_mali))
    # print('P~:', len(df_mali)-len(data_P))
    print(len(columns_list),len(feature_list),data_P.shape[1])

    return data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list


# def generate_balance_pudata(seed,data_path,label_frequency,method = 'auto'):
#     #method = {'auto','smote',}

xgb = XGBClassifier(n_estimators=350, learning_rate=0.25, max_depth=6, n_jobs=-1, random_state=2018)
#svm = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True, degree=3, random_state=2018)
svm = SVC(C=1.0, kernel='linear', gamma='auto', probability=True, degree=3, random_state=2018)
rf = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018)
lr = LogisticRegression(penalty='l2', C=1.0, solver='sag', max_iter=1000, random_state=2019, n_jobs=-1)
# solver lbfgs

def performance_contrast(seed,data_path,label_frequency,key=0):
    if key==0:
        print("Wrong key value,evaluate wrong feature list!")
        return
    start = time.time()
    data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list = generate_pudata(seed,data_path,label_frequency)
    # # PUMD_RF
    # append =1
    # selection = 2
    # original = 3
    mode = 'run'
    lf_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #定制化修改参数
    beta_list_fs = [0.9, 1.0, 1.0, 1.1, 1.0, 1.1, 1.3, 1.3, 1.9]
    beta_list_org = [0.9, 1.0, 1.0, 1.1, 1.0, 1.2, 1.3, 1.3, 1.9]
    beta_list_del = [2.0, 2.5, 4.5, 5.0, 5.5, 8.0, 6.5, 9.0, 9.5]

    ad_para_org = [19]*9
    ad_para_del = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0]
    ad_para_fs = [19]*9

    contamination_org = 0.046
    contamination_del = 0.046
    contamination_fs = 0.046
    # 定制化修改参数 -- end
    beta_dict_fs = dict(zip(lf_list, beta_list_fs))
    beta_dict_org = dict(zip(lf_list, beta_list_org))
    beta_dict_del = dict(zip(lf_list, beta_list_del))
    beta_dict = [0,beta_dict_org,beta_dict_fs,beta_dict_del]
    para_dict_fs = dict(zip(lf_list, ad_para_fs))
    para_dict_org = dict(zip(lf_list, ad_para_org))
    para_dict_del = dict(zip(lf_list, ad_para_del))
    ad_para = [0,para_dict_org,para_dict_fs,para_dict_del]
    contamination = [0,contamination_org ,contamination_fs,contamination_del ]





    pumd_rf = PUMD(data_P, data_U, rf, return_proba=True, pred_U=pred_U, feature_list=feature_list, mode=mode,
                   ad_para=ad_para[key][label_frequency], contamination=contamination[key], ad_beta=beta_dict[key][label_frequency])
    y_pred, y_prob = pumd_rf.predict()

    pumd_rf_performance = model_performance(y_pred, pred_U, y_prob)
    print('PUMD_rf_performance', pumd_rf_performance)
    # print(pumd_rf.__repr__())


    decription = 'The evaluation of the algorithm has been completed.'
    print(decription, 'Running_Time:{:.2f}s\n'.format(time.time()-start))
    df_value = [label_frequency]
    df_value.extend(pumd_rf_performance)
    return df_value




if __name__ == '__main__':
    data_path_org= r"SRC_df_label/df_original.csv "
    data_path_append = r"SRC_df_label/df_append.csv"
    data_path_selection = r"SRC_df_label/df_selection.csv"

    # 定制化修改参数

    label_frequency = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df_org = pd.DataFrame(columns = ['lf','f_score', 'gmean', 'recall','precision', 'acc','auc','mcc'])
    df_fs = pd.DataFrame(columns=['lf', 'f_score', 'gmean', 'recall', 'precision', 'acc', 'auc', 'mcc'])
    df_del = pd.DataFrame(columns=['lf', 'f_score', 'gmean', 'recall', 'precision', 'acc', 'auc', 'mcc'])
    for i, c in enumerate(label_frequency):
        print('-------------------------label_frequency:{}'.format(c))
        df_org.loc[i] = performance_contrast(100, data_path_append, c,1)
        df_fs.loc[i] = performance_contrast(100,  data_path_selection, c,2)
        df_del.loc[i] = performance_contrast(100,  data_path_org, c,3)
    now = time.time()
    df_org.to_csv(r'Final_result\org_append_selection\\' + str(now)+'pumd_profile_dl_append_fl.csv',index=False)
    df_fs.to_csv(r'Final_result\org_append_selection\\' +str(now)+ 'pumd_profile_dl_selection_fl.csv', index=False)
    df_del.to_csv(r'Final_result\org_append_selection\\' + str(now)+'pumd_profile_dl_org_fl.csv', index=False)



