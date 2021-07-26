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
import math
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
    print(len(np_fs))
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
    # print(len(columns_list),len(feature_list),data_P.shape[1])

    return data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list



xgb = XGBClassifier(n_estimators=350, learning_rate=0.25, max_depth=6, n_jobs=-1, random_state=2018)
#svm = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True, degree=3, random_state=2018)
svm = SVC(C=1.0, kernel='linear', gamma='auto', probability=True, degree=3, random_state=2018)
rf = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018)
lr = LogisticRegression(penalty='l2', C=1.0, solver='sag', max_iter=1000, random_state=2019, n_jobs=-1)
# solver lbfgs

def performance_contrast(seed,data_path,label_frequency):

    df_value = []
    df_f_score, df_gmean, df_recall, df_precision, df_acc, df_auc, df_mcc = [label_frequency], [label_frequency], [label_frequency], [label_frequency], [label_frequency], [label_frequency], [label_frequency]
    start = time.time()
    data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list = generate_pudata(seed,data_path,label_frequency)


    #-----------------------------------model_evaluate--------------------------------------
    mode = 'run'

    ## [1]PUMD
    ### RF
    lf_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    beta_list = [0.9,1.0,1.0,1.1,1.0,1.1,1.3,1.3,1.9]
    beta_dict = dict(zip(lf_list, beta_list))
    pumd_rf = PUMD(data_P, data_U, rf, return_proba=True, pred_U=pred_U, feature_list=feature_list, mode=mode,
                   ad_para=19, contamination=0.046, ad_beta=beta_dict[label_frequency])

    y_pred, y_prob = pumd_rf.predict()
    pumd_rf_performance = model_performance(y_pred, pred_U, y_prob)
    print('pumd_rf_performance:', pumd_rf_performance)
    df_f_score.append(pumd_rf_performance[0])
    df_gmean.append(pumd_rf_performance[1])
    df_recall.append(pumd_rf_performance[2])
    df_precision.append(pumd_rf_performance[3])
    df_acc.append(pumd_rf_performance[4])
    df_auc.append(pumd_rf_performance[5])
    df_mcc.append(pumd_rf_performance[6])


    ### xgb
    pumd_xgb = PUMD(data_P, data_U, xgb, return_proba=True, pred_U=pred_U, feature_list=feature_list, mode=mode,
                   ad_para=19, contamination=0.046, ad_beta=beta_dict[label_frequency])
    y_pred, y_prob = pumd_xgb.predict()
    pumd_xgb_performance = model_performance(y_pred, pred_U, y_prob)
    print('pumd_xgb_performance:', pumd_xgb_performance)
    df_f_score.append(pumd_xgb_performance[0])
    df_gmean.append(pumd_xgb_performance[1])
    df_recall.append(pumd_xgb_performance[2])
    df_precision.append(pumd_xgb_performance[3])
    df_acc.append(pumd_xgb_performance[4])
    df_auc.append(pumd_xgb_performance[5])
    df_mcc.append(pumd_xgb_performance[6])
    ### svm
    pumd_svm = PUMD(data_P, data_U, svm, return_proba=True, pred_U=pred_U, feature_list=feature_list, mode=mode,
                       ad_para=19, contamination=0.046, ad_beta=beta_dict[label_frequency])
    y_pred, y_prob = pumd_svm.predict()
    pumd_svm_performance = model_performance(y_pred, pred_U, y_prob)
    print('pumd_svm_performance:', pumd_svm_performance)
    df_f_score.append(pumd_svm_performance[0])
    df_gmean.append(pumd_svm_performance[1])
    df_recall.append(pumd_svm_performance[2])
    df_precision.append(pumd_svm_performance[3])
    df_acc.append(pumd_svm_performance[4])
    df_auc.append(pumd_svm_performance[5])
    df_mcc.append(pumd_svm_performance[6])


    ## lr
    pumd_lr =PUMD(data_P, data_U, lr, return_proba=True, pred_U=pred_U, feature_list=feature_list, mode=mode,
                       ad_para=19, contamination=0.046, ad_beta=beta_dict[label_frequency])
    y_pred, y_prob = pumd_lr.predict()
    pumd_lr_performance = model_performance(y_pred, pred_U, y_prob)
    print('pumd_lr_performance :', pumd_lr_performance )
    df_f_score.append(pumd_lr_performance [0])
    df_gmean.append(pumd_lr_performance [1])
    df_recall.append(pumd_lr_performance [2])
    df_precision.append(pumd_lr_performance [3])
    df_acc.append(pumd_lr_performance [4])
    df_auc.append(pumd_lr_performance [5])
    df_mcc.append(pumd_lr_performance [6])


    decription = 'The evaluation of the algorithm has been completed.'
    print(decription, 'Running_Time:{:.2f}s\n'.format(time.time()-start))

    df_value.append(df_f_score)
    df_value.append(df_gmean)

    df_value.append(df_recall)
    df_value.append(df_precision)
    df_value.append(df_acc)
    df_value.append(df_auc)
    df_value.append(df_mcc)

    return df_value




if __name__ == '__main__':

    data_path =r"SRC_df_label/df_selection.csv"
    label_frequency = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    columns_list = col = ['label_frequency', 'PUMD_rf', 'PUMD_xgboost', 'PUMD_svm', 'PUMD_lr']
    # [f_score, gmean, recall,precision, acc,auc,mcc]
    df_evaluate_fscore = pd.DataFrame(columns=columns_list)
    df_evaluate_gmean = pd.DataFrame(
        columns=columns_list)
    df_evaluate_recall = pd.DataFrame(
        columns=columns_list)
    df_evaluate_precision = pd.DataFrame(
        columns=columns_list)
    df_evaluate_acc = pd.DataFrame(
        columns=columns_list)
    df_evaluate_auc = pd.DataFrame(
        columns=columns_list)
    df_evaluate_mcc = pd.DataFrame(
        columns=columns_list)
    for num, c in enumerate(label_frequency):
        print('-------------label_frequency is {}------------------------------:'.format(c))
        df_value = performance_contrast(100, data_path, c)
        # [f_score, gmean, recall,precision, acc,auc,mcc]
        df_evaluate_fscore.loc[num] = df_value[0]
        df_evaluate_gmean.loc[num] = df_value[1]
        df_evaluate_recall.loc[num] = df_value[2]
        df_evaluate_precision.loc[num] = df_value[3]
        df_evaluate_acc.loc[num] = df_value[4]
        df_evaluate_auc.loc[num] = df_value[5]
        df_evaluate_mcc.loc[num] = df_value[6]
    now = time.time()

    df_evaluate_fscore.to_csv(r'Final_result\model_evaluate\\'  + str(now)+'-fscore_model_evaluate.csv',
                              index=False)
    df_evaluate_gmean.to_csv(r'Final_result\model_evaluate\\' + str(now)+'-gmean_model_evaluate.csv',
                             index=False)
    df_evaluate_recall.to_csv(r'Final_result\model_evaluate\\'+str(now)+ '-recall_model_evaluate.csv',
                              index=False)
    df_evaluate_precision.to_csv(r'Final_result\model_evaluate\\'+ str(now)+'-precision_model_evaluate.csv',
                                 index=False)

    df_evaluate_acc.to_csv(r'Final_result\model_evaluate\\'+ str(now)+'-acc_model_evaluate.csv',
                           index=False)
    df_evaluate_auc.to_csv(r'Final_result\model_evaluate\\'  +str(now)+'-auc_model_evaluate.csv',
                           index=False)
    df_evaluate_mcc.to_csv(r'Final_result\model_evaluate\\' +str(now)+ '-mcc_model_evaluate.csv',
                           index=False)


